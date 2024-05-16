include("../../src/PertTheory.jl")
include("./params.jl")

gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0]
let
    g = gs[parse(Int, ARGS[1])]
    params = (J,g,S)
    obs = obs_Schw
    F   = F_Schw

    data = []

    Threads.@threads for num in 1:N
        @show num
        t = 0
        dataTemp = []

        a0    = mu .* exp.(1im * sigma * 2pi * randn(L[2]))
        aDag0 = mu .* exp.(1im * sigma * 2pi * randn(L[2]))
        fields = (a0, aDag0)

        # Compute time evolution
        try
            while t < T
                push!(dataTemp, [t, obs(fields, params)...])
                fields = heun_step(fields, params, F, dt)
                # dt_new, fields = adaptive_heun_step(fields, params, F, dt, 1e-1)
                t += dt
            end
        catch
            @warn "Error, fields = $fields"
            continue
        end
        push!(data, dataTemp)
    end

    df_res = analyze_data(data, params)
    CSV.write("../../data/TWA_Schw_L=$(L[2])_Sz=$(S)_N=$(N)_g=$(g)_sigma=$(sigma).csv", df_res)
end;
