import NCDatasets
using Printf




function returnTanInverseInProperQuadrant(DeltaX, DeltaY; printAngle = false)
    if DeltaX !== 0.0
        if DeltaX > 0.0 && DeltaY > 0.0 # First Quadrant
            angle = atan(DeltaY / DeltaX)
        elseif DeltaX < 0.0 && DeltaY > 0.0 # Second Quadrant
            angle = pi + atan(DeltaY / DeltaX)
        elseif DeltaX < 0.0 && DeltaY < 0.0 # Third Quadrant
            angle = pi + atan(DeltaY / DeltaX)
        elseif DeltaX > 0.0 && DeltaY < 0.0 # Fourth Quadrant
            angle = 2.0 * pi + atan(DeltaY / DeltaX)
        elseif DeltaX > 0.0 && DeltaY == 0.0
            angle = 0.0
        elseif DeltaX < 0.0 && DeltaY == 0.0
            angle = pi
        end
    else
        if DeltaY > 0.0
            angle = pi / 2.0
        elseif DeltaY < 0.0
            angle = -pi / 2.0
        else
            print("DeltaX = 0 && DeltaY = 0! Stopping!")
            return
        end
    end

    if printAngle
        if DeltaX !== 0.0
            @sprintf("DeltaY/DeltaX = %.15f.", (DeltaY / DeltaX))
        end
        @sprintf("The angle in radians is %.15f.", angle)
        @sprintf("The angle in degrees is %.15f.", (angle * 180.0 / pi))
        if DeltaX !== 0.0
            @sprintf("The trigonometric tangent of the angle is %.15f.", tan(angle))
        end
    end

    return angle
end



function fix_angleEdge(
    mpasOcean;
    determineYCellAlongLatitude = true,
    printOutput = false,
    printRelevantMeshData = false,
)

    return fix_angleEdge(
        dcEdge,
        xCell,
        yCell,
        xEdge,
        yEdge,
        angleEdge,
        cellsOnEdge,
        nCells,
        determineYCellAlongLatitude,
        printOutput,
        printRelevantMeshData,
    )

end

function fix_angleEdge(
    dcEdge,
    xCell,
    yCell,
    xEdge,
    yEdge,
    angleEdge,
    cellsOnEdge,
    nCells;
    determineYCellAlongLatitude = true,
    printOutput = false,
    printRelevantMeshData = false,
)

    sqrt3over2 = sqrt(3.0) / 2.0

    # cwd = pwd()
    # path = mesh_directory # cwd * "/" * mesh_directory * "/"
    #     if ! isdir(path)
    #         mkpath(path) # os.makedir(path)
    #     end
    # cd(path)
    # mesh_file = NCDatasets.Dataset(mesh_file_name, "r", format=:netcdf4_classic)


    dcEdge = round(maximum(dcEdge))
    DeltaXMax = maximum(dcEdge) * 1.1
    xCell::Array{Float64,1} = xCell
    yCell::Array{Float64,1} = yCell
    # println("yCell ", sizeof(yCell))
    nCells = length(yCell)
    # The determination of yCellAlongLatitude in the following lines only holds for rectangular structured meshes
    # with equal number of cells in each direction. However, for a problem with non-periodic boundary conditions,
    # it will work for the culled mesh && the final mesh, but not the base mesh.

    if false #determineYCellAlongLatitude
        nY = Int64(round(sqrt(nCells)))
        yCellAlongLatitude = zeros(Float32, nY)
        iYAlongLatitude = 0 + 1
        for iY in range(0 + 1, nCells, step = 1)
            if mod(Float64(iY), Float64(nY)) == 0.0
                yCellAlongLatitude[iYAlongLatitude] = yCell[iY]
                iYAlongLatitude += 1
            end
        end
        # println("final iYAL $iYAlongLatitude, ny (size ycal) $(nY)")
        DeltaYMax = maximum(diff(yCellAlongLatitude))
        # println("DeltaYMax $DeltaYMax, DeltaXMax $DeltaXMax, approx $(DeltaXMax*sqrt3over2)")
    else
        DeltaYMax = DeltaXMax * sqrt3over2 * 2#+100
    end
    xEdge = xEdge
    yEdge = yEdge

    angleEdge = angleEdge

    cellsOnEdge = cellsOnEdge
    nEdges = length(angleEdge)
    nCells = nCells
    computed_angleEdge = zeros(nEdges)
    tolerance = 1e-3

    # DeltaXs = zeros(Float64, nEdges)
    # DeltaYs = zeros(Float64, nEdges)

    for iEdge in range(1, nEdges, step = 1)
        thisXEdge = xEdge[iEdge]
        thisYEdge = yEdge[iEdge]
        cell1 = cellsOnEdge[1, iEdge]
        cell2 = cellsOnEdge[2, iEdge]

        xCell1 = xCell[cell1]
        yCell1 = yCell[cell1]


        if cell2 == 0
            if thisXEdge > xCell1 && abs(thisYEdge - yCell1) < tolerance
                DeltaX = dcEdge
                DeltaY = 0.0
            elseif thisXEdge > xCell1 && thisYEdge > yCell1
                DeltaX = dcEdge / 2.0
                DeltaY = sqrt3over2 * dcEdge
            elseif thisXEdge > xCell1 && thisYEdge < yCell1
                DeltaX = dcEdge / 2.0
                DeltaY = -sqrt3over2 * dcEdge
            elseif thisXEdge < xCell1 && abs(thisYEdge - yCell1) < tolerance
                DeltaX = -dcEdge
                DeltaY = 0.0
            elseif thisXEdge < xCell1 && thisYEdge > yCell1
                DeltaX = -dcEdge / 2.0
                DeltaY = sqrt3over2 * dcEdge
            elseif thisXEdge < xCell1 && thisYEdge < yCell1
                DeltaX = -dcEdge / 2.0
                DeltaY = -sqrt3over2 * dcEdge
            end
        else
            xCell2 = xCell[cell2]
            DeltaX = xCell2 - xCell1
            yCell2 = yCell[cell2]
            DeltaY = yCell2 - yCell1
            if abs(DeltaY) < tolerance && DeltaX < 0.0 && abs(DeltaX) > DeltaXMax
                # cells [{4,1},{8,5},{12,9},{16,13}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge
            elseif abs(DeltaY) < tolerance && DeltaX > 0.0 && abs(DeltaX) > DeltaXMax
                # cells [{1,4},{5,8},{9,12},{13,16}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge
            elseif DeltaX < 0.0 &&
                   DeltaY < 0.0 &&
                   abs(DeltaX) > DeltaXMax &&
                   abs(DeltaY) > DeltaYMax
                # cells [{16,1}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge / 2.0
                DeltaY = sqrt3over2 * dcEdge
            elseif DeltaX < 0.0 &&
                   DeltaY < 0.0 &&
                   abs(DeltaX) > DeltaXMax &&
                   abs(DeltaY) <= DeltaYMax
                # cells [{8,1},{16,9}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge / 2.0
                DeltaY = -sqrt3over2 * dcEdge
            elseif DeltaX < 0.0 &&
                   DeltaY < 0.0 &&
                   abs(DeltaX) < DeltaXMax &&
                   abs(DeltaY) > DeltaYMax
                # cells [{13,1},{14,2},{15,3},{16,4}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge / 2.0
                DeltaY = sqrt3over2 * dcEdge
            elseif DeltaX < 0.0 &&
                   DeltaY > 0.0 &&
                   abs(DeltaX) < DeltaXMax &&
                   abs(DeltaY) > DeltaYMax
                # cells [{2,13},{3,14},{4,15}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge / 2.0
                DeltaY = -sqrt3over2 * dcEdge
            elseif DeltaX < 0.0 &&
                   DeltaY > 0.0 &&
                   abs(DeltaX) > DeltaXMax &&
                   abs(DeltaY) <= DeltaYMax
                # cells [{8,9}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge / 2.0
                DeltaY = sqrt3over2 * dcEdge
            elseif DeltaX > 0.0 &&
                   DeltaY < 0.0 &&
                   abs(DeltaX) > DeltaXMax &&
                   abs(DeltaY) <= DeltaYMax
                # cells [{9,8}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge / 2.0
                DeltaY = -sqrt3over2 * dcEdge
            elseif DeltaX > 0.0 &&
                   DeltaY < 0.0 &&
                   abs(DeltaX) < DeltaXMax &&
                   abs(DeltaY) > DeltaYMax
                # cells [{13,2},{14,3},{15,4}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge / 2.0
                DeltaY = sqrt3over2 * dcEdge
            elseif DeltaX > 0.0 &&
                   DeltaY > 0.0 &&
                   abs(DeltaX) < DeltaXMax &&
                   abs(DeltaY) > DeltaYMax
                # cells [{1,13},{2,14},{3,15},{4,16}] for a regular structured 4 x 4 mesh
                DeltaX = dcEdge / 2.0
                DeltaY = -sqrt3over2 * dcEdge
            elseif DeltaX > 0.0 &&
                   DeltaY > 0.0 &&
                   abs(DeltaX) > DeltaXMax &&
                   abs(DeltaY) <= DeltaYMax
                # cells [{1,8},{9,16}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge / 2.0
                DeltaY = sqrt3over2 * dcEdge
            elseif DeltaX > 0.0 &&
                   DeltaY > 0.0 &&
                   abs(DeltaX) > DeltaXMax &&
                   abs(DeltaY) > DeltaYMax
                # cells [{1,16}] for a regular structured 4 x 4 mesh
                DeltaX = -dcEdge / 2.0
                DeltaY = -sqrt3over2 * dcEdge
            end
        end
        if DeltaX == 0 && DeltaY == 0
            println("dx dy 0 iEdge: $iEdge mpascellsonedge: $(cellsOnEdge[:,iEdge])")
        end
        computed_angleEdge[iEdge] = returnTanInverseInProperQuadrant(DeltaX, DeltaY)
        if printOutput
            # printOutput should be specified as True only for small meshes consisting of 4 x 4 cells.
            if printRelevantMeshData
                # printRelevantMeshData should be specified as True only for small meshes consisting of 4 x 4 cells.
                println(
                    "%2d [%2d %2d] %+9.2f [%+9.2f %+9.2f] %+9.2f %+9.2f [%+9.2f %+9.2f] %+8.2f [%+5.2f %+5.2f]",
                    (
                        iEdge,
                        cell1,
                        cell2,
                        thisXEdge,
                        xCell1,
                        xCell2,
                        DeltaX,
                        thisYEdge,
                        yCell1,
                        yCell2,
                        DeltaY,
                        angleEdge[iEdge],
                        computed_angleEdge[iEdge],
                    ),
                )
            else
                println(
                    "For edge %2d with cellsOnEdge = {%2d,%2d}, {angleEdge, computed_angleEdge} is {%.2f, %.2f}.",
                    (iEdge + 1, cell1, cell2, angleEdge[iEdge], computed_angleEdge[iEdge]),
                )
            end
        end
    end
    # cd(cwd)
    return computed_angleEdge
end
