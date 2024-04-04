import NCDatasets
using SparseArrays
using Adapt
using KernelAbstractions
#using CUDA

include("fixAngleEdge.jl")

const KA = KernelAbstractions

mutable struct MPAS_Ocean{FT}

    # prognostic variables
    normalVelocityCurrent::Array{FT,2}    # group velocity normal to mesh edges (edge-centered)
    normalVelocityTendency::Array{FT,2}    # tendency (edge-centered)


    layerThickness::Array{FT,2}
    layerThicknessTendency::Array{FT,2}


    layerThicknessEdge::Array{FT,2}

    # (ssh used to be prognostic for single-layer, now each has its own layerthickness
    sshCurrent::Array{FT,1}    # sea surface height (cell-centered)
#     sshTendency::Array{FT,1}    # tendency (cell-centered)



    bottomDepth::Array{FT,1}   # bathymetry (cell-centered)
    bottomDepthEdge::Array{FT,1}    # bathymetry (cell-centered)
    gravity::FT
    nVertLevels::Int64
    dt::FT



    ### mesh information


    ## cell-centered arrays
    nCells::Int64  # number of cells
    cellsOnCell::Array{Int64,2}  # indices of neighbor cells (nCells, nEdgesOnCell[Int64])
    edgesOnCell::Array{Int64,2}  # indices of edges of cell (nCells, nEdgesOnCell[Int64])
    verticesOnCell::Array{Int64,2}  # indices of vertices of cell (nCells, nEdgesOnCell[Int64])
    kiteIndexOnCell::Array{Int64,2}  # index of kite shape on cell connecting vertex, midpoint of edge, center of cell, and midpoint of other edge (nCells, nEdgesOnCell[Int64])
    nEdgesOnCell::Array{Int64,1}  # number of edges a cell has (cell-centered)
    edgeSignOnCell::Array{Int8,2} # orientation of edge relative to cell (nCells, nEdgesOnCell[Int64])
    latCell::Array{FT,1}    # latitude of cell (cell-centered)
    lonCell::Array{FT,1}    # longitude of cell (cell-centered)
    xCell::Array{FT,1}    # x coordinate of cell (cell-centered)
    yCell::Array{FT,1}    # y coordinate of cell (cell-centered)
    areaCell::Array{FT,1}    # area of cell (cell-centered)
    fCell::Array{FT,1}    # coriolis parameter (cell-centered)
    maxLevelCell::Array{Int64,1}
    gridSpacing::Array{FT,1}
    boundaryCell::Array{Int64,2}  # 0 for inner cells, 1 for boundary cells (cell-centered)
    cellMask::Array{Int64,2}


    ## edge-centered arrays
    nEdges::Int64
    cellsOnEdge::Array{Int64,2}  # indices of the two cells on this edge (nEdges, 2)
    edgesOnEdge::Array{Int64,2}
    verticesOnEdge::Array{Int64,2}
    nEdgesOnEdge::Array{Int64,1}
    xEdge::Array{FT,1}
    yEdge::Array{FT,1}
    dvEdge::Array{FT,1}
    dcEdge::Array{FT,1}
    fEdge::Array{FT,1}    # coriolis parameter
    angleEdge::Array{FT,1}
    weightsOnEdge::Array{FT,2}    # coeffecients of norm vels of surrounding edges in linear combination to compute tangential velocity
    maxLevelEdgeTop::Array{Int64,1}
    maxLevelEdgeBot::Array{Int64,1}
    boundaryEdge::Array{Int64,2}
    edgeMask::Array{Int64,2}


    ## vertex-centered arrays
    nVertices::Int64
    latVertex::Array{FT,1}
    lonVertex::Array{FT,1}
    xVertex::Array{FT,1}
    yVertex::Array{FT,1}
    vertexDegree::Int64
    cellsOnVertex::Array{Int64}
    edgesOnVertex::Array{Int64}
    edgeSignOnVertex::Array{Int8}
    fVertex::Array{FT}
    areaTriangle::Array{FT}
    kiteAreasOnVertex::Array{FT}
    maxLevelVertexTop::Array{Int64}
    maxLevelVertexBot::Array{Int64}
    boundaryVertex::Array{Int64,2}
    vertexMask::Array{Int64,2}


    gridSpacingMagnitude::FT

    # maximum values of x and y (size of mesh)
    lX::FT
    lY::FT

    nNonPeriodicBoundaryCells::Int64
    nNonPeriodicBoundaryEdges::Int64
    nNonPeriodicBoundaryVertices::Int64

    backend::KA.Backend

end
    function MPAS_Ocean{FT}(mesh_directory = "MPAS_O_Shallow_Water/Mesh+Initial_Condition+Registry_Files/Periodic",
                        base_mesh_file_name = "base_mesh.nc",
                        mesh_file_name = "mesh.nc";
                        nvlevels = 1,
                        periodicity = "Periodic",
                        cells = "All",
                        backend=CPU()) where {FT}
                        #backend=CUDABackend()) where {FT}

        #mpasOcean = new{FT}()
        base_mesh_file = NCDatasets.Dataset("$mesh_directory/$base_mesh_file_name", "r", format=:netcdf4)
        mesh_file = NCDatasets.Dataset("$mesh_directory/$mesh_file_name", "r", format=:netcdf4)

        # Choose my_mesh_file_name to be either base_mesh_file_name or mesh_file_name
        my_mesh_file_name = mesh_file_name
        # Choose my_mesh_file to be either base_mesh_file or mesh_file
        my_mesh_file = mesh_file


        maxEdges = mesh_file.dim["maxEdges"]

        ## defining the mesh
        nGlobalCells = mesh_file.dim["nCells"]
        nGlobalEdges = mesh_file.dim["nEdges"]
        nGlobalVertices = mesh_file.dim["nVertices"]
        _vertexDegree = mesh_file.dim["vertexDegree"]
        _nVertLevels = nvlevels

        # will be incremented in ocn_init_routines_compute_max_level
        _nNonPeriodicBoundaryEdges = 0
        _nNonPeriodicBoundaryVertices = 0
        _nNonPeriodicBoundaryCells = 0



        _edgesOnCell = my_mesh_file["edgesOnCell"]
        _verticesOnCell = my_mesh_file["verticesOnCell"]

        if cells == "All"
            cells = 1:nGlobalCells
        end
        edges = collect(Set(_edgesOnCell[:,cells]))
        vertices = collect(Set(_verticesOnCell[:,cells]))

        _nCells = nCells = length(cells)
        _nVertices = nVertices = length(vertices)
        _nEdges = nEdges = length(edges)

        _cellsOnCell = my_mesh_file["cellsOnCell"][:,cells]#, globtolocalcell)
        _edgesOnCell = my_mesh_file["edgesOnCell"][:,cells]#, globtolocaledge)
        _verticesOnCell = my_mesh_file["verticesOnCell"][:,cells]#, globtolocalvertex)


        _cellsOnEdge = my_mesh_file["cellsOnEdge"][:,edges]# :: Array{Int64,2}#, globtolocalcell)
        _edgesOnEdge = my_mesh_file["edgesOnEdge"][:,edges]# :: Array{Int64,2}
        _verticesOnEdge = my_mesh_file["verticesOnEdge"][:,edges]# :: Array{Int64,2}

        _cellsOnVertex = my_mesh_file["cellsOnVertex"][:,vertices]
        _edgesOnVertex = my_mesh_file["edgesOnVertex"][:,vertices]

        #########################
        # change the mesh array's indexes to point to local indexes if this is a partition of a full simulation
        #ordermapEOE, ordermapCOV = localizeIfPartition!(mpasOcean, cells, edges, vertices) # this returns maps we use to reorder other arrays
        ############BEGIN#############
        """
        if this mpas ocean is a partition of a larger ocean (like when the ocean is distributed among nodes),
        we need to go through the mesh-arrays and change the index fields so they point to the index within
        the current partition (local) rather than the index within the whole simulation (global)
        """

        # change each index in a set of indexes to local based on globmap - dict map from global to local index
        function globalToLocal!(globals, globmap, sort=false)
            for i in 1:length(globals)
                if globals[i] in keys(globmap)
                    globals[i] = globmap[globals[i]]
                else
                    # this index points to something outside this partition (or outside the entire boundary)
                    globals[i] = 0 # our code is built to handle an index of 0 as something that doesn't exist
                end
            end

            if sort
                # move "0" values to the end of each row, like a ragged array, like the code expects
                ordermap = dimsortperm(globals, dims=1, rev=true)
                globals[:] = globals[ordermap]
                # its now necesary to remember the way we reordered the indexes, and reorder related arrays the same way
                return ordermap # so we return the array that applies this reordering
            end
        end

        # create maps from global to local indices, by "reversing" array describing which global indexes to use
        globtolocalcell = Dict{Int64,Int64}()
        for (iLocal, iGlobal) in enumerate(cells)
            globtolocalcell[iGlobal] = iLocal
        end

        globtolocaledge = Dict{Int64,Int64}()
        for (iLocal, iGlobal) in enumerate(edges)
            globtolocaledge[iGlobal] = iLocal
        end

        globtolocalvertex = Dict{Int64,Int64}()
        for (iLocal, iGlobal) in enumerate(vertices)
            globtolocalvertex[iGlobal] = iLocal
        end

        # apply these maps to adapt global indexes to local indexes in mesh info fields
        globalToLocal!(_edgesOnCell, globtolocaledge)# = my_mesh_file["edgesOnCell"][:][:,cells]#, globtolocaledge)
        globalToLocal!(_cellsOnCell, globtolocalcell)# = my_mesh_file["cellsOnCell"][:][:,cells]#, globtolocalcell)
        globalToLocal!(_verticesOnCell, globtolocalvertex)# = my_mesh_file["verticesOnCell"][:][:,cells]#, globtolocalvertex)

        globalToLocal!(_cellsOnEdge, globtolocalcell, true)# = my_mesh_file["cellsOnEdge"][:][:,edges]#, globtolocalcell)
        # the weightsOnEdge[:,:] array is supposed to correspond to the edgesOnEdge[:,:] array, so we need to reorder it the same way we reorder this, by keeping the map we use to reorder.
        ordermapEOE = globalToLocal!(_edgesOnEdge, globtolocaledge, true)# = my_mesh_file["edgesOnEdge"][:][:,edges], globtolocaledge)
        globalToLocal!(_verticesOnEdge, globtolocalvertex)# = globalToLocal(my_mesh_file["verticesOnEdge"][:][:,edges], globtolocalvertex)

        # the kiteAreasOnVertex[:,:] array is supposed to correspond to the cellsOnVertex[:,:] array, so we need to reorder it the same way we reorder this, by keeping the map we use to reorder.
        ordermapCOV = globalToLocal!(_cellsOnVertex, globtolocalcell, true)# = globalToLocal(my_mesh_file["cellsOnVertex"][:][:,vertices], globtolocalcell)
        globalToLocal!(_edgesOnVertex, globtolocaledge, true)#


        # nEdgesOnEdge may be different now if some connected edges are outside this partition
        # so we need to tell it how many valid (non-zero, contained in this partition) edges there are
        _nEdgesOnCell = countnonzero(_edgesOnCell, dims=1) #base_mesh_file["nEdgesOnCell"][:][cells]
        _nEdgesOnEdge = countnonzero(_edgesOnEdge, dims=1) #base_mesh_file["nEdgesOnCell"][:][cells]
        ############END#############
        _xEdge = my_mesh_file["xEdge"][:][edges]
        _yEdge = my_mesh_file["yEdge"][:][edges]


        _latCell = base_mesh_file["latCell"][:][cells]
        _lonCell = base_mesh_file["lonCell"][:][cells]
        _xCell = base_mesh_file["xCell"][:][cells]
        _yCell = base_mesh_file["yCell"][:][cells]
        _areaCell = my_mesh_file["areaCell"][:][cells]

        _dvEdge = my_mesh_file["dvEdge"][:][edges]
        _dcEdge = my_mesh_file["dcEdge"][:][edges]

        _kiteAreasOnVertex = my_mesh_file["kiteAreasOnVertex"][:,vertices][ordermapCOV]
        _areaTriangle = my_mesh_file["areaTriangle"][:][vertices]

        _latVertex = base_mesh_file["latVertex"][:][vertices]
        _lonVertex = base_mesh_file["lonVertex"][:][vertices]
        _xVertex = base_mesh_file["xVertex"][:][vertices]
        _yVertex = base_mesh_file["yVertex"][:][vertices]

        # NCDatasets.load!(NCDatasets.variable(my_mesh_file,"weightsOnEdge"),mpasOcean.weightsOnEdge,:,:)[:,edges]
        _weightsOnEdge = my_mesh_file["weightsOnEdge"][:,edges][ordermapEOE]


    #         mpasOcean.gridSpacing = mesh_file["gridSpacing"][:]
        _gridSpacing = sqrt.(_areaCell)#mpasOcean.xCell[2] - mpasOcean.xCell[1]

        useGridSpacingMagnitudeDefaultDefinition = true
        if useGridSpacingMagnitudeDefaultDefinition
            _gridSpacingMagnitude = _gridSpacing[0+1]
        else
            _gridSpacingMagnitude = _xCell[1+1] - _xCell[0+1]
        end


        # For a mesh with non-periodic zonal boundaries, adjust the zonal coordinates of the cell centers,
        # vertices and edges.
        dx = _gridSpacingMagnitude # Int64.e. dx = mpasOcean.dcEdge[0]
        dy = sqrt(3.0)/2.0*dx
        if periodicity == "NonPeriodic_x" || periodicity == "NonPeriodic_xy"
            _xCell[:] .-= dx
            _xVertex[:] .-= dx
            _xEdge[:] .-= dx
        end
        if periodicity == "NonPeriodic_y" || periodicity == "NonPeriodic_xy"
            _yCell[:] .-= dy
            _yVertex[:] .-= dy
            _yEdge[:] .-= dy
        end

        # Specify the zonal && meridional extents of the domain.
        _lX = round(maximum(_xCell)) # Int64.e. mpasOcean.lX = sqrt(float(mpasOcean.nCells))*dx
        # Please note that for all of our test problems, the MPAS-O mesh is generated in such a way that
        # mpasOcean.lX Int64.e. the number of cells in the x (|| y) direction times dx has zero fractional part in
        # units of m, for which we can afford to round it to attain perfection.
        _lY = sqrt(3.0)/2.0*_lX # Int64.e. mpasOcean.lY = max(_yVertex)

                _angleEdge = my_mesh_file["angleEdge"][:][edges]
                # if true
                #_angleEdge[:] = fix_angleEdge(mpasOcean,determineYCellAlongLatitude=true,
                #                           printOutput=false,printRelevantMeshData=false)
                _angleEdge[:] = fix_angleEdge(_dcEdge, _xCell, _yCell,
                                              _xEdge, _yEdge, _angleEdge,
                                              _cellsOnEdge, _nCells,
                                              determineYCellAlongLatitude=true,
                                              printOutput=false,printRelevantMeshData=false)
                # end

        # Define && initialize the following arrays not contained within either the base mesh file || the mesh
        # file.
        _fVertex = zeros(FT, nVertices)
        _fCell = zeros(FT, nCells)
        _fEdge = zeros(FT, nEdges)
        _bottomDepth = zeros(FT, nCells)
        _bottomDepthEdge = zeros(FT, nEdges)

        _layerThickness = zeros(FT, (_nVertLevels, nCells))
        _layerThicknessTendency = zeros(FT, (_nVertLevels, nCells))
        _layerThicknessEdge = zeros(FT, (_nVertLevels, nEdges))


        # define coriolis parameter and bottom depth.
        # currently hard-wired to these values, uniform everywhere for linear case
        f0 = 0.0001
        _fCell[:] .= f0
        _fEdge[:] .= f0
        _fVertex[:] .= f0

        H = 1000.0
        _bottomDepth[:] .= H
        _bottomDepthEdge[:] .= H # if not flat: change to average of bottomDepth of cells on either side

        # initialize layer thicknesses as even division of total depth with level surface
        for k in 1:_nVertLevels
            _layerThickness[k,:] .= _bottomDepth[:] ./ _nVertLevels
            _layerThicknessEdge[k,:] .= _bottomDepthEdge[:] ./ _nVertLevels
        end


        _kiteIndexOnCell = zeros(Int64, (nCells,maxEdges))
        _edgeSignOnVertex = zeros(Int8, (nVertices,maxEdges))

        _edgeSignOnCell = zeros(Int8, (nCells,maxEdges))

        #########################
        #ocn_init_routines_setup_sign_and_index_fields!(mpasOcean)
        ############BEGIN#############
        for iCell in range(1,_nCells,step=1)
            for j in range(1,_nEdgesOnCell[iCell],step=1)
                iEdge = _edgesOnCell[j,iCell]
                iVertex = _verticesOnCell[j,iCell]
                # Vector points from cell 1 to cell 2
                if _cellsOnEdge[1,iEdge] == iCell
                    _edgeSignOnCell[iCell,j] = -1
                else
                    _edgeSignOnCell[iCell,j] = 1
                end
                for jj in range(1,_vertexDegree,step=1)
                    if _cellsOnVertex[jj,iVertex] == iCell
                        _kiteIndexOnCell[iCell,j] = jj
                    end
                end
            end
        end


        for iVertex in range(1,_nVertices,step=1)
            for j in range(1,_vertexDegree,step=1)
                iEdge = _edgesOnVertex[j,iVertex]
                if iEdge != 0
                    # Vector points from vertex 1 to vertex 2
                    if _verticesOnEdge[1,iEdge] == iVertex
                        _edgeSignOnVertex[iVertex,j] = -1
                    else
                        _edgeSignOnVertex[iVertex,j] = 1
                    end
                end
            end
        end
        ############END#############


        _boundaryCell = zeros(Int64, (nCells, _nVertLevels))
        _boundaryEdge = zeros(Int64, (nEdges, _nVertLevels))
        _boundaryVertex = zeros(Int64, (nVertices, _nVertLevels))
        _cellMask = zeros(Int64, (nCells, _nVertLevels))
        _edgeMask = zeros(Int64, (nEdges, _nVertLevels))
        _vertexMask = zeros(Int64, (nVertices, _nVertLevels))

        _maxLevelCell = zeros(Int64, nCells)
        _maxLevelEdgeTop = zeros(Int64, nEdges)
        _maxLevelEdgeBot = zeros(Int64, nEdges)
        _maxLevelVertexTop = zeros(Int64, nVertices)
        _maxLevelVertexBot = zeros(Int64, nVertices)


        ## defining the prognostic variables

        _normalVelocityCurrent = zeros(FT, (_nVertLevels, nEdges))
        _normalVelocityTendency = zeros(FT, (_nVertLevels, nEdges))


        # ssh no longer prognostic, replaced with layerThickness
        _sshCurrent = zeros(FT, nCells)
#         mpasOcean.sshTendency = zeros(FT, nCells)

        _gravity = 9.8

        # calculate minimum dt based on CFL condition
        courantNumber = 0.4
        _dt = courantNumber * minimum(_dcEdge) / sqrt(_gravity * maximum(_bottomDepth))
        #########################
        #ocn_init_routines_compute_max_level!(mpasOcean)
        ############BEGIN#############
        _maxLevelCell[:] .= _nVertLevels

        for iEdge in 1:_nEdges
            iCell1 = _cellsOnEdge[1,iEdge]
            iCell2 = _cellsOnEdge[2,iEdge]
            if iCell1 == 0 || iCell2 == 0
                _boundaryEdge[iEdge,:] .= 1.0
                _maxLevelEdgeTop[iEdge] = 0
                if iCell1 == 0
                    _maxLevelEdgeBot[iEdge] = _maxLevelCell[iCell2]
                elseif iCell2 == 0
                    _maxLevelEdgeBot[iEdge] = _maxLevelCell[iCell1]
                end
            elseif ! ( iCell1 == 0 || iCell2 == 0 )
                _maxLevelEdgeTop[iEdge] = min(_maxLevelCell[iCell1], _maxLevelCell[iCell2])
                _maxLevelEdgeBot[iEdge] = max(_maxLevelCell[iCell1], _maxLevelCell[iCell2])
            end
        end

        for iVertex in 1:_nVertices
            iCell1 = _cellsOnVertex[1,iVertex]
            if iCell1 == 0
                _maxLevelVertexBot[iVertex] = -1
                _maxLevelVertexTop[iVertex] = -1
            else
                _maxLevelVertexBot[iVertex] = _maxLevelCell[iCell1]
                _maxLevelVertexTop[iVertex] = _maxLevelCell[iCell1]
            end

            for Int64 in 1:_vertexDegree
                iCell = _cellsOnVertex[Int64, iVertex]
                if iCell == 0
                    _maxLevelVertexBot[iVertex] = max(_maxLevelVertexBot[iVertex], -1)
                    _maxLevelVertexBot[iVertex] = min(_maxLevelVertexTop[iVertex], -1)
                else
                    _maxLevelVertexBot[iVertex] = max(_maxLevelVertexBot[iVertex],
                                                                _maxLevelCell[iCell])
                    _maxLevelVertexTop[iVertex] = min(_maxLevelVertexTop[iVertex],
                                                                _maxLevelCell[iCell])
                end
            end
        end

        # determine_boundaryEdge_Generalized_Method = true
        #
        # if determine_boundaryEdge_Generalized_Method
        #     mpasOcean.boundaryEdge[:,:] .= 1
        # end
        _edgeMask[:,:] .= 0

        for iEdge in 1:_nEdges
            index_UpperLimit = _maxLevelEdgeTop[iEdge]
            if index_UpperLimit > -1
    #             if determine_boundaryEdge_Generalized_Method
    #                 mpasOcean.boundaryEdge[iEdge,1:index_UpperLimit+1] .= 0
    #             end
                _edgeMask[iEdge,1:index_UpperLimit] .= 1
            end
        end

        for iEdge in 1:_nEdges
            iCell1 = _cellsOnEdge[1, iEdge]
            iCell2 = _cellsOnEdge[2, iEdge]

            iVertex1 = _verticesOnEdge[1, iEdge]
            iVertex2 = _verticesOnEdge[2, iEdge]

            if _boundaryEdge[iEdge] == 1
                if iCell1 != 0
                    _boundaryCell[iCell1] = 1
                end
                if iCell2 != 0
                    _boundaryCell[iCell2] = 1
                end
                _boundaryVertex[iVertex1] = 1
                _boundaryVertex[iVertex2] = 1
            end
        end


        for iCell in 1:_nCells
            for k in 1:_nVertLevels
                if _maxLevelCell[iCell] >= k
                    _cellMask[iCell, k] = 1
                end
            end
        end
        for iVertex in 1:_nVertices
            for k in 1:_nVertLevels
                if _maxLevelVertexBot[iVertex] >= k
                    _vertexMask[iVertex, k] = 1
                end
            end
        end

        _nNonPeriodicBoundaryEdges = 0.0
        for iEdge in 1:_nEdges
            if _boundaryEdge[iEdge, 1] == 1.0
                _nNonPeriodicBoundaryEdges += 1
            end
        end

        _nNonPeriodicBoundaryVertices = 0.0
        for iVertex in 1:_nVertices
            if _boundaryVertex[iVertex, 1] == 1.0
                _nNonPeriodicBoundaryVertices += 1
            end
        end

        _nNonPeriodicBoundaryCells = 0.0
        for iCell in 1:_nCells
            if _boundaryCell[iCell, 1] == 1.0
                _nNonPeriodicBoundaryCells += 1
            end
        end
        ############END#############
        return MPAS_Ocean{FT}(
            adapt(backend, _normalVelocityCurrent),
            adapt(backend, _normalVelocityTendency),
            adapt(backend, _layerThickness),
            adapt(backend, _layerThicknessTendency),
            adapt(backend, _layerThicknessEdge),
            adapt(backend, _sshCurrent),
            adapt(backend, _bottomDepth),
            adapt(backend, _bottomDepthEdge),
            adapt(backend, _gravity),
            _nVertLevels,
            adapt(backend, _dt),
            _nCells,
            _cellsOnCell,
            _edgesOnCell,
            _verticesOnCell,
            _kiteIndexOnCell,
            _nEdgesOnCell,
            _edgeSignOnCell,
            adapt(backend, _latCell),
            adapt(backend, _lonCell),
            adapt(backend, _xCell),
            adapt(backend, _yCell),
            adapt(backend, _areaCell),
            adapt(backend, _fCell),
            _maxLevelCell,
            adapt(backend, _gridSpacing),
            _boundaryCell,
            _cellMask,
            _nEdges,
            _cellsOnEdge,
            _edgesOnEdge,
            _verticesOnEdge,
            _nEdgesOnEdge,
            adapt(backend, _xEdge),
            adapt(backend, _yEdge),
            adapt(backend, _dvEdge),
            adapt(backend, _dcEdge),
            adapt(backend, _fEdge),
            adapt(backend, _angleEdge),
            adapt(backend, _weightsOnEdge),
            _maxLevelEdgeTop,
            _maxLevelEdgeBot,
            _boundaryEdge,
            _edgeMask,
            _nVertices,
            adapt(backend, _latVertex),
            adapt(backend, _lonVertex),
            adapt(backend, _xVertex),
            adapt(backend, _yVertex),
            _vertexDegree,
            _cellsOnVertex,
            _edgesOnVertex,
            _edgeSignOnVertex,
            adapt(backend, _fVertex),
            adapt(backend, _areaTriangle),
            adapt(backend, _kiteAreasOnVertex),
            _maxLevelVertexTop,
            _maxLevelVertexBot,
            _boundaryVertex,
            _vertexMask,
            adapt(backend, _gridSpacingMagnitude),
            adapt(backend, _lX),
            adapt(backend, _lY),
            _nNonPeriodicBoundaryCells,
            _nNonPeriodicBoundaryEdges,
            _nNonPeriodicBoundaryVertices,
            backend
            )
    end
#end


function Adapt.adapt(mpasOcean::MPAS_Ocean, backend=CPU()) #backend=CUDABackend())
    return MPAS_Ocean(
        adapt(backend, mpasOcean.normalVelocityCurrent),
        adapt(backend, mpasOcean.normalVelocityTendency),
        adapt(backend, mpasOcean.layerThickness),
        adapt(backend, mpasOcean.layerThicknessTendency),
        adapt(backend, mpasOcean.layerThicknessEdge),
        adapt(backend, mpasOcean.sshCurrent),
        adapt(backend, mpasOcean.bottomDepth),
        adapt(backend, mpasOcean.bottomDepthEdge),
        adapt(backend, mpasOcean.gravity),
        mpasOcean.nVertLevels,
        adapt(backend, mpasOcean.dt),
        mpasOcean.nCells,
        mpasOcean.cellsOnCell,
        mpasOcean.edgesOnCell,
        mpasOcean.verticesOnCell,
        mpasOcean.kiteIndexOnCell,
        mpasOcean.nEdgesOnCell,
        mpasOcean.edgeSignOnCell,
        adapt(backend, mpasOcean.latCell),
        adapt(backend, mpasOcean.lonCell),
        adapt(backend, mpasOcean.xCell),
        adapt(backend, mpasOcean.yCell),
        adapt(backend, mpasOcean.areaCell),
        adapt(backend, mpasOcean.fCell),
        mpasOcean.maxLevelCell,
        adapt(backend, mpasOcean.gridSpacing),
        mpasOcean.boundaryCell,
        mpasOcean.cellMask,
        mpasOcean.nEdges,
        mpasOcean.cellsOnEdge,
        mpasOcean.edgesOnEdge,
        mpasOcean.verticesOnEdge,
        mpasOcean.nEdgesOnEdge,
        adapt(backend, mpasOcean.xEdge),
        adapt(backend, mpasOcean.yEdge),
        adapt(backend, mpasOcean.dvEdge),
        adapt(backend, mpasOcean.dcEdge),
        adapt(backend, mpasOcean.fEdge),
        adapt(backend, mpasOcean.angleEdge),
        adapt(backend, mpasOcean.weightsOnEdge),
        mpasOcean.maxLevelEdgeTop,
        mpasOcean.maxLevelEdgeBot,
        mpasOcean.boundaryEdge,
        mpasOcean.edgeMask,
        mpasOcean.nVertices,
        adapt(backend, mpasOcean.latVertex),
        adapt(backend, mpasOcean.lonVertex),
        adapt(backend, mpasOcean.xVertex),
        adapt(backend, mpasOcean.yVertex),
        mpasOcean.vertexDegree,
        mpasOcean.cellsOnVertex,
        mpasOcean.edgesOnVertex,
        mpasOcean.edgeSignOnVertex,
        adapt(backend, mpasOcean.fVertex),
        adapt(backend, mpasOcean.areaTriangle),
        adapt(backend, mpasOcean.kiteAreasOnVertex),
        mpasOcean.maxLevelVertexTop,
        mpasOcean.maxLevelVertexBot,
        mpasOcean.boundaryVertex,
        mpasOcean.vertexMask,
        adapt(backend, mpasOcean.gridSpacingMagnitude),
        adapt(backend, mpasOcean.lX),
        adapt(backend, mpasOcean.lY),
        mpasOcean.nNonPeriodicBoundaryCells,
        mpasOcean.nNonPeriodicBoundaryEdges,
        mpasOcean.nNonPeriodicBoundaryVertices,
        backend
        )
end



function dimsortperm(A::AbstractMatrix; dims::Integer, rev::Bool = false)
    """
        function used to sort (and remember the sorted order) of array in speified dimensions only
        (essentially just extension of builtin sortperm function)
        credit to stevengj on issue: https://github.com/JuliaLang/julia/issues/16273#issuecomment-228787141
    """
    P = mapslices(x -> sortperm(x; rev = rev), A, dims = dims)
    if dims == 1
        for j = 1:size(P, 2)
            offset = (j - 1) * size(P, 1)
            for i = 1:size(P, 1)
                P[i, j] += offset
            end
        end
    else # if dims == 2
        for j = 1:size(P, 2)
            for i = 1:size(P, 1)
                P[i, j] = (P[i, j] - 1) * size(P, 1) + i
            end
        end
    end
    return P
end

function countnonzero(A; dims=1)
    """
        count nonzero elements in each row (or col depending on value of dims)
    """
    return dropdims(sum(map(val -> Int(val != 0), A), dims=dims), dims=dims)
end

function localizeIfPartition!(mpasOcean, cells, edges, vertices)
    """
        if this mpas ocean is a partition of a larger ocean (like when the ocean is distributed among nodes),
        we need to go through the mesh-arrays and change the index fields so they point to the index within
        the current partition (local) rather than the index within the whole simulation (global)
    """

    # change each index in a set of indexes to local based on globmap - dict map from global to local index
    function globalToLocal!(globals, globmap, sort=false)
        for i in 1:length(globals)
            if globals[i] in keys(globmap)
                globals[i] = globmap[globals[i]]
            else
                # this index points to something outside this partition (or outside the entire boundary)
                globals[i] = 0 # our code is built to handle an index of 0 as something that doesn't exist
            end
        end

        if sort
            # move "0" values to the end of each row, like a ragged array, like the code expects
            ordermap = dimsortperm(globals, dims=1, rev=true)
            globals[:] = globals[ordermap]
            # its now necesary to remember the way we reordered the indexes, and reorder related arrays the same way
            return ordermap # so we return the array that applies this reordering
        end
    end

    # create maps from global to local indices, by "reversing" array describing which global indexes to use
    globtolocalcell = Dict{Int64,Int64}()
    for (iLocal, iGlobal) in enumerate(cells)
        globtolocalcell[iGlobal] = iLocal
    end

    globtolocaledge = Dict{Int64,Int64}()
    for (iLocal, iGlobal) in enumerate(edges)
        globtolocaledge[iGlobal] = iLocal
    end

    globtolocalvertex = Dict{Int64,Int64}()
    for (iLocal, iGlobal) in enumerate(vertices)
        globtolocalvertex[iGlobal] = iLocal
    end

    # apply these maps to adapt global indexes to local indexes in mesh info fields
    globalToLocal!(mpasOcean.edgesOnCell, globtolocaledge)# = my_mesh_file["edgesOnCell"][:][:,cells]#, globtolocaledge)
    globalToLocal!(mpasOcean.cellsOnCell, globtolocalcell)# = my_mesh_file["cellsOnCell"][:][:,cells]#, globtolocalcell)
    globalToLocal!(mpasOcean.verticesOnCell, globtolocalvertex)# = my_mesh_file["verticesOnCell"][:][:,cells]#, globtolocalvertex)

    globalToLocal!(mpasOcean.cellsOnEdge, globtolocalcell, true)# = my_mesh_file["cellsOnEdge"][:][:,edges]#, globtolocalcell)
    # the weightsOnEdge[:,:] array is supposed to correspond to the edgesOnEdge[:,:] array, so we need to reorder it the same way we reorder this, by keeping the map we use to reorder.
    ordermapEOE = globalToLocal!(mpasOcean.edgesOnEdge, globtolocaledge, true)# = my_mesh_file["edgesOnEdge"][:][:,edges], globtolocaledge)
    globalToLocal!(mpasOcean.verticesOnEdge, globtolocalvertex)# = globalToLocal(my_mesh_file["verticesOnEdge"][:][:,edges], globtolocalvertex)

    # the kiteAreasOnVertex[:,:] array is supposed to correspond to the cellsOnVertex[:,:] array, so we need to reorder it the same way we reorder this, by keeping the map we use to reorder.
    ordermapCOV = globalToLocal!(mpasOcean.cellsOnVertex, globtolocalcell, true)# = globalToLocal(my_mesh_file["cellsOnVertex"][:][:,vertices], globtolocalcell)
    globalToLocal!(mpasOcean.edgesOnVertex, globtolocaledge, true)#


    # nEdgesOnEdge may be different now if some connected edges are outside this partition
    # so we need to tell it how many valid (non-zero, contained in this partition) edges there are
    mpasOcean.nEdgesOnCell = countnonzero(mpasOcean.edgesOnCell, dims=1) #base_mesh_file["nEdgesOnCell"][:][cells]
    mpasOcean.nEdgesOnEdge = countnonzero(mpasOcean.edgesOnEdge, dims=1) #base_mesh_file["nEdgesOnCell"][:][cells]

    # send back needed maps to reorder sister arrays identically
    return ordermapEOE, ordermapCOV
end




function ocn_init_routines_setup_sign_and_index_fields!(mpasOcean)
    for iCell in range(1,mpasOcean.nCells,step=1)
        for j in range(1,mpasOcean.nEdgesOnCell[iCell],step=1)
            iEdge = mpasOcean.edgesOnCell[j,iCell]
            iVertex = mpasOcean.verticesOnCell[j,iCell]
            # Vector points from cell 1 to cell 2
            if mpasOcean.cellsOnEdge[1,iEdge] == iCell
                mpasOcean.edgeSignOnCell[iCell,j] = -1
            else
                mpasOcean.edgeSignOnCell[iCell,j] = 1
            end
            for jj in range(1,mpasOcean.vertexDegree,step=1)
                if mpasOcean.cellsOnVertex[jj,iVertex] == iCell
                    mpasOcean.kiteIndexOnCell[iCell,j] = jj
                end
            end
        end
    end


    for iVertex in range(1,mpasOcean.nVertices,step=1)
        for j in range(1,mpasOcean.vertexDegree,step=1)
            iEdge = mpasOcean.edgesOnVertex[j,iVertex]
            if iEdge != 0
                # Vector points from vertex 1 to vertex 2
                if mpasOcean.verticesOnEdge[1,iEdge] == iVertex
                    mpasOcean.edgeSignOnVertex[iVertex,j] = -1
                else
                    mpasOcean.edgeSignOnVertex[iVertex,j] = 1
                end
            end
        end
    end
end



function ocn_init_routines_compute_max_level!(mpasOcean)
    mpasOcean.maxLevelCell[:] .= mpasOcean.nVertLevels

    for iEdge in 1:mpasOcean.nEdges
        iCell1 = mpasOcean.cellsOnEdge[1,iEdge]
        iCell2 = mpasOcean.cellsOnEdge[2,iEdge]
        if iCell1 == 0 || iCell2 == 0
            mpasOcean.boundaryEdge[iEdge,:] .= 1.0
            mpasOcean.maxLevelEdgeTop[iEdge] = 0
            if iCell1 == 0
                mpasOcean.maxLevelEdgeBot[iEdge] = mpasOcean.maxLevelCell[iCell2]
            elseif iCell2 == 0
                mpasOcean.maxLevelEdgeBot[iEdge] = mpasOcean.maxLevelCell[iCell1]
            end
        elseif ! ( iCell1 == 0 || iCell2 == 0 )
            mpasOcean.maxLevelEdgeTop[iEdge] = min(mpasOcean.maxLevelCell[iCell1], mpasOcean.maxLevelCell[iCell2])
            mpasOcean.maxLevelEdgeBot[iEdge] = max(mpasOcean.maxLevelCell[iCell1], mpasOcean.maxLevelCell[iCell2])
        end
    end

    for iVertex in 1:mpasOcean.nVertices
        iCell1 = mpasOcean.cellsOnVertex[1,iVertex]
        if iCell1 == 0
            mpasOcean.maxLevelVertexBot[iVertex] = -1
            mpasOcean.maxLevelVertexTop[iVertex] = -1
        else
            mpasOcean.maxLevelVertexBot[iVertex] = mpasOcean.maxLevelCell[iCell1]
            mpasOcean.maxLevelVertexTop[iVertex] = mpasOcean.maxLevelCell[iCell1]
        end

        for Int64 in 1:mpasOcean.vertexDegree
            iCell = mpasOcean.cellsOnVertex[Int64, iVertex]
            if iCell == 0
                mpasOcean.maxLevelVertexBot[iVertex] = max(mpasOcean.maxLevelVertexBot[iVertex], -1)
                mpasOcean.maxLevelVertexBot[iVertex] = min(mpasOcean.maxLevelVertexTop[iVertex], -1)
            else
                mpasOcean.maxLevelVertexBot[iVertex] = max(mpasOcean.maxLevelVertexBot[iVertex],
                                                            mpasOcean.maxLevelCell[iCell])
                mpasOcean.maxLevelVertexTop[iVertex] = min(mpasOcean.maxLevelVertexTop[iVertex],
                                                            mpasOcean.maxLevelCell[iCell])
            end
        end
    end

    # determine_boundaryEdge_Generalized_Method = true
    #
    # if determine_boundaryEdge_Generalized_Method
    #     mpasOcean.boundaryEdge[:,:] .= 1
    # end
    mpasOcean.edgeMask[:,:] .= 0

    for iEdge in 1:mpasOcean.nEdges
        index_UpperLimit = mpasOcean.maxLevelEdgeTop[iEdge]
        if index_UpperLimit > -1
#             if determine_boundaryEdge_Generalized_Method
#                 mpasOcean.boundaryEdge[iEdge,1:index_UpperLimit+1] .= 0
#             end
            mpasOcean.edgeMask[iEdge,1:index_UpperLimit] .= 1
        end
    end

    for iEdge in 1:mpasOcean.nEdges
        iCell1 = mpasOcean.cellsOnEdge[1, iEdge]
        iCell2 = mpasOcean.cellsOnEdge[2, iEdge]

        iVertex1 = mpasOcean.verticesOnEdge[1, iEdge]
        iVertex2 = mpasOcean.verticesOnEdge[2, iEdge]

        if mpasOcean.boundaryEdge[iEdge] == 1
            if iCell1 != 0
                mpasOcean.boundaryCell[iCell1] = 1
            end
            if iCell2 != 0
                mpasOcean.boundaryCell[iCell2] = 1
            end
            mpasOcean.boundaryVertex[iVertex1] = 1
            mpasOcean.boundaryVertex[iVertex2] = 1
        end
    end


    for iCell in 1:mpasOcean.nCells
        for k in 1:mpasOcean.nVertLevels
            if mpasOcean.maxLevelCell[iCell] >= k
                mpasOcean.cellMask[iCell, k] = 1
            end
        end
    end
    for iVertex in 1:mpasOcean.nVertices
        for k in 1:mpasOcean.nVertLevels
            if mpasOcean.maxLevelVertexBot[iVertex] >= k
                mpasOcean.vertexMask[iVertex, k] = 1
            end
        end
    end

    mpasOcean.nNonPeriodicBoundaryEdges = 0.0
    for iEdge in 1:mpasOcean.nEdges
        if mpasOcean.boundaryEdge[iEdge, 1] == 1.0
            mpasOcean.nNonPeriodicBoundaryEdges += 1
        end
    end

    mpasOcean.nNonPeriodicBoundaryVertices = 0.0
    for iVertex in 1:mpasOcean.nVertices
        if mpasOcean.boundaryVertex[iVertex, 1] == 1.0
            mpasOcean.nNonPeriodicBoundaryVertices += 1
        end
    end

    mpasOcean.nNonPeriodicBoundaryCells = 0.0
    for iCell in 1:mpasOcean.nCells
        if mpasOcean.boundaryCell[iCell, 1] == 1.0
            mpasOcean.nNonPeriodicBoundaryCells += 1
        end
    end
end
