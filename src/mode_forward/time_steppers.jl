
function forward_backward_step!(mpasOcean)
    calculate_normal_velocity_tendency!(mpasOcean)

    update_normal_velocity_by_tendency!(mpasOcean)

    calculate_thickness_tendency!(mpasOcean)

    update_thickness_by_tendency!(mpasOcean)
end

function forward_euler_step!(mpasOcean)
    calculate_normal_velocity_tendency!(mpasOcean)

    calculate_thickness_tendency!(mpasOcean)

    update_normal_velocity_by_tendency!(mpasOcean)

    update_thickness_by_tendency!(mpasOcean)
end
