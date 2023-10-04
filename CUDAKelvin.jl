CODE_ROOT = pwd() * "/"

include(CODE_ROOT * "mode_init/MPAS_Ocean.jl")
include(CODE_ROOT * "mode_init/exactsolutions.jl")
include(CODE_ROOT * "mode_init/MPAS_Ocean_CUDA.jl")
include(CODE_ROOT * "mode_forward/time_steppers_cuda.jl")


using Enzyme
using LinearAlgebra # for norm()

function zeroseed(shadowmpasOceanCuda)
    fields=fieldnames(typeof(shadowmpasOceanCuda));
    for i=1:length(fields)
        value=getfield(shadowmpasOceanCuda, fields[i])
        if(isa(value, Array))
            value.=zero(eltype(value));
        elseif(isa(value, CUDA.CuArray))
            value.=zero(eltype(value));
        else
            value=zero(eltype(value));
        end
        setfield!(shadowmpasOceanCuda,fields[i],value)
    end
    return nothing
end

function kelvin_test(mesh_directory, base_mesh_file_name, mesh_file_name, periodicity, T, dt, nSaves=1;
    plot=false, animate=false, nvlevels=1)
    mpasOcean = MPAS_Ocean(mesh_directory,base_mesh_file_name,mesh_file_name, periodicity=periodicity, nvlevels=nvlevels)

    meanFluidThicknessH = sum(mpasOcean.bottomDepth)/length(mpasOcean.bottomDepth)
    c = sqrt(mpasOcean.gravity*meanFluidThicknessH)

    println("simulating for T: $T")
    lYedge = maximum(mpasOcean.yEdge) - minimum(mpasOcean.yEdge)

    function lateralProfilePeriodic(y)
        return 1e-6*cos(y/mpasOcean.lY * 4 * pi)
    end

    period = lYedge / (4*pi) /c

    lateralProfile = lateralProfilePeriodic

    println("generating kelvin wave exact methods for mesh")
    kelvinWaveExactNormalVelocity, kelvinWaveExactSSH, kelvinWaveExactSolution!, boundaryCondition! = kelvinWaveGenerator(mpasOcean, lateralProfile)

    println("setting up initial condition")
    kelvinWaveExactSolution!(mpasOcean)
    
    println("original dt $(mpasOcean.dt)")
    nSteps = Int(round(T/mpasOcean.dt/nSaves))
    mpasOcean.dt = T / nSteps / nSaves 

    println("dx $(mpasOcean.dcEdge[1]) \t dt $(mpasOcean.dt) \t dx/c $(maximum(mpasOcean.dcEdge) / c) \t dx/dt $(mpasOcean.dcEdge[1]/mpasOcean.dt)")
    println("period $period \t steps $nSteps")

    mpasOceanCuda = MPAS_Ocean_CUDA(mpasOcean)
    function timesteps(mpasOceanCuda::MPAS_Ocean_CUDA)
        #t = 0.0
        for i in 1:nSaves
            for j in 1:nSteps
                forward_backward_step_cuda!(mpasOceanCuda)
            end
        end
        return nothing
    end

    shadowmpasOceanCuda = deepcopy(mpasOceanCuda)
    zeroseed(shadowmpasOceanCuda)
    shadowmpasOceanCuda.normalVelocityCurrent .= 1.0
    shadowmpasOceanCuda.layerThickness .= 1.0
    Enzyme.autodiff_deferred(Reverse, timesteps, Const, Duplicated(mpasOceanCuda, shadowmpasOceanCuda))
    #timesteps(mpasOceanCuda)
    println("normalVelocityCurrent ", shadowmpasOceanCuda.normalVelocityCurrent)
    println("layerThickness ", shadowmpasOceanCuda.layerThickness)

    return nothing
end


T = 7000
nCellsX = 64
kelvin_test(
    CODE_ROOT * "MPAS_Ocean_Shallow_Water_Meshes/CoastalKelvinWaveMesh/ConvergenceStudyMeshes",
    "culled_mesh_$(nCellsX)x$(nCellsX).nc", "mesh_$(nCellsX)x$(nCellsX).nc", "NonPeriodic_x", T, 75, 2, plot=true, animate=true, nvlevels=1)

