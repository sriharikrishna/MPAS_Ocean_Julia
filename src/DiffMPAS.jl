module DiffMPAS
include("mode_init/MPAS_Ocean.jl")
include("mode_forward/calculate_tendencies.jl")
include("mode_forward/time_steppers.jl")
#include(CODE_ROOT * "visualization.jl")
include("mode_init/exactsolutions.jl")

export MPAS_Ocean,
    kelvinWaveExactSolution!,
    kevlinWaveExactSSH,
    kelvinWaveExactNormalVelocity,
    kelvinWaveGenerator,
    seed,
    kelvin_test
export forward_backward_step!
end # module DiffMPAS
