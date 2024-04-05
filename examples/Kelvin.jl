using DiffMPAS

using Adapt
using Enzyme
using LinearAlgebra # for norm()
using LazyArtifacts
using Test
using CUDA
using KernelAbstractions

const DATA = joinpath(artifact"MPASData")

function seed(shadowmpasOcean)
    fields = fieldnames(typeof(shadowmpasOcean))
    for i = 1:length(fields)-1
        value = getfield(shadowmpasOcean, fields[i])
        if (isa(value, Array))
            value .= zero(eltype(value))
        else
            value = zero(eltype(value))
        end
        setfield!(shadowmpasOcean, fields[i], value)
    end
    return nothing
end

function timesteps(mpasOcean::MPAS_Ocean)
    #t = 0.0
    for i = 1:mpasOcean.nSaves
        for j = 1:mpasOcean.nSteps
            forward_backward_step!(mpasOcean)
        end
    end
    return nothing
end

function kelvin_test(
    mesh_directory,
    base_mesh_file_name,
    mesh_file_name,
    periodicity,
    T,
    dt,
    nSaves = 1;
    plot = false,
    animate = false,
    nvlevels = 1,
    backend = CPU(),
)
    mpasOcean = MPAS_Ocean{Float64}(
        mesh_directory,
        base_mesh_file_name,
        mesh_file_name,
        periodicity = periodicity,
        nvlevels = nvlevels,
    )

    meanFluidThicknessH = sum(mpasOcean.bottomDepth) / length(mpasOcean.bottomDepth)
    c = sqrt(mpasOcean.gravity * meanFluidThicknessH)

    println("simulating for T: $T")
    lYedge = maximum(mpasOcean.yEdge) - minimum(mpasOcean.yEdge)

    function lateralProfilePeriodic(y)
        return 1e-6 * cos(y / mpasOcean.lY * 4 * pi)
    end

    period = lYedge / (4 * pi) / c

    lateralProfile = lateralProfilePeriodic

    println("generating kelvin wave exact methods for mesh")
    kelvinWaveExactNormalVelocity,
    kelvinWaveExactSSH,
    kelvinWaveExactSolution!,
    boundaryCondition! = kelvinWaveGenerator(mpasOcean, lateralProfile)

    println("setting up initial condition")
    kelvinWaveExactSolution!(mpasOcean)

    sshOverTimeNumerical = zeros(Float64, (mpasOcean.nCells, nSaves))
    sshOverTimeExact = zeros(Float64, (mpasOcean.nCells, nSaves))
    nvOverTimeNumerical = zeros(Float64, (mpasOcean.nEdges, nSaves))
    nvOverTimeExact = zeros(Float64, (mpasOcean.nEdges, nSaves))

    sshOverTimeNumerical[:, 1] =
        dropdims(sum(mpasOcean.layerThickness, dims = 1), dims = 1) - mpasOcean.bottomDepth
    sshOverTimeExact[:, 1] = kelvinWaveExactSSH(mpasOcean, 1:mpasOcean.nCells)
    nvOverTimeNumerical[:, 1] = sum(mpasOcean.normalVelocityCurrent, dims = 1)
    nvOverTimeExact[:, 1] = kelvinWaveExactNormalVelocity(mpasOcean, 1:mpasOcean.nEdges)

    println("original dt $(mpasOcean.dt)")
    nSteps = Int(round(T / mpasOcean.dt / nSaves))
    mpasOcean.nSteps = nSteps
    mpasOcean.nSaves = nSaves
    nSaves = nSaves
    mpasOcean.dt = T / nSteps / nSaves

    println(
        "dx $(mpasOcean.dcEdge[1]) \t dt $(mpasOcean.dt) \t dx/c $(maximum(mpasOcean.dcEdge) / c) \t dx/dt $(mpasOcean.dcEdge[1]/mpasOcean.dt)",
    )
    println("period $period \t steps $nSteps")

    mpasOcean = adapt(backend, mpasOcean)

    timesteps(mpasOcean)

    shadowmpasOcean = deepcopy(mpasOcean)
    CUDA.allowscalar() do
        # seed(shadowmpasOcean)
    end
    timesteps(mpasOcean)
    # if backend == CPU()
    #     Enzyme.autodiff(Reverse, timesteps, Const, Duplicated(mpasOcean, shadowmpasOcean))
    # end

    error = sshOverTimeNumerical .- sshOverTimeExact
    MaxErrorNorm = norm(error, Inf)
    L2ErrorNorm = norm(error / sqrt(float(mpasOcean.nCells)))

    return mpasOcean, MaxErrorNorm, L2ErrorNorm
end

backends = [CPU(), CUDABackend()]
@testset "Test $backend Forward" for backend in backends
    T = 7000
    nCellsX = 64
    mpasocean, MaxErrorNorm, L2ErrorNorm = kelvin_test(
        DATA,
        "culled_mesh_$(nCellsX)x$(nCellsX).nc",
        "mesh_$(nCellsX)x$(nCellsX).nc",
        "NonPeriodic_x",
        T,
        75,
        2,
        plot = true,
        animate = true,
        nvlevels = 1,
        backend=backend,
    )
    @testset "Tests" begin
        @test isapprox(MaxErrorNorm, 0.0, atol = 1e-12)
        @test isapprox(L2ErrorNorm, 0.0, atol = 1e-12)
    end
    println("nCells: $(mpasocean.nCells)")
    println("dt: $(mpasocean.dt)")
end
