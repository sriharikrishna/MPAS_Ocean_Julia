include("calculate_tendencies_ka.jl")

######## KernelAbstractions versions of time-advancing  methods


function forward_backward_step_ka!(mpasOcean)
    calculate_normal_velocity_tendency_ka!(mpasOcean)

    update_normal_velocity_by_tendency_ka!(mpasOcean)

    calculate_thickness_tendency_ka!(mpasOcean)

    update_thickness_by_tendency_ka!(mpasOcean)
end

function forward_euler_step_ka!(mpasOcean)
    calculate_normal_velocity_tendency_ka!(mpasOcean)

    calculate_thickness_tendency_ka!(mpasOcean)

    update_normal_velocity_by_tendency_ka!(mpasOcean)

    update_thickness_by_tendency_ka!(mpasOcean)
end
