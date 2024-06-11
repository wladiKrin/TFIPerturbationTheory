function sgn(x)
    return x == 0 ? Int(0) : Int(x/abs(x))
end

function laguerre(x::T, N::Int) where T
    p0, p1 = one(T), -x+1
    N == 0 && return p0
    for k = one(T):N-1
        p1, p0 = ((2k+1)/(k+1) - x/(k+1))*p1 - k/(k+1)*p0, p1
    end
    p1
end

#   function get_samples(N::Int; n_samples::Int = 1e5, dx::Float64 = 1e-3)
#       fact = N
#       domain = (0,2.5)
#
#       f(x)     = 2*fact*x*exp.(-fact*x) * laguerre(2*fact*x, N)
#       f_abs(x) = abs(f(x))
#       f_abs_mean(x) = sgn(f(x)) * x * f_abs(x)
#       f_mean(x) = x * f(x)
#       
#       # f_var(x) = x^2 * f(x)
#       
#       points = first(domain):dx:last(domain)
#
#       f_points = f.(collect(points))
#       f_abs_points = abs.(f_points)
#       sum_f_abs_points = sum(f_abs_points)
#
#       cum_f_abs = cumsum(f_abs_points ./ sum_f_abs_points)
#
#       probs = rand(n_samples)
#
#       samples = [findmin(abs.(cum_f_abs .- u))[2]*dx for u in probs]
#       signs = sgn.(f.(samples))
#       prefactor = sum_f_abs_points / sum(f_points)
#
#       return samples, signs, prefactor
#   end

function get_samples(N::Int; n_samples::Int = 1e5, dx::Float64 = 1e-3)
    fact = N

    f(x)     = 2*fact*x*exp.(-fact*x) * laguerre(2*fact*x, N)
    f_mean(x) = x * f(x)
    # f_abs(x) = abs(f(x))
    # f_abs_mean(x) = sgn(f(x)) * x * f_abs(x)
    
    domain = (0,2.5)
    points = collect(first(domain):dx:last(domain))

    f_points      = f.(points)
    f_mean_points = f_mean.(points)
    mean_f        = sum(f_mean_points) / sum(f_points)

    domain = (0,1.5)
    points = collect(first(domain):dx:last(domain))

    g(x)     = f(x * mean_f)
    g_abs(x) = abs(g(x))
    g_abs_mean(x) = sgn(g(x)) * x * g_abs(x)

    g_points      = g.(points)
    g_abs_points = abs.(g_points)
    sum_g_abs_points = sum(g_abs_points)

    cum_g_abs = cumsum(g_abs_points ./ sum_g_abs_points)

    probs = rand(n_samples)

    samples = [findmin(abs.(cum_g_abs .- u))[2]*dx for u in probs]
    signs = sgn.(g.(samples))
    prefactor = sum_g_abs_points / sum(g_points)

    return samples, signs, prefactor
end

##############################################################
############### Truncated Wigner Approximation ###############
##############################################################

# Schwinger
function F_Schw(fields, params)
    J, g, S = params
    a, b = fields

    F = map(1:length(a)) do index
        ai   = a[index]
        ai1  = index+1 > length(a) ? a[1] : a[index+1]
        ai_1 = index-1 < 1 ? a[end] : a[index-1]

        bi   = b[index]
        bi1  = index+1 > length(b) ? b[1] : b[index+1]
        bi_1 = index-1 < 1 ? b[end] : b[index-1]


        f1 = abs2(ai) - abs2(ai1)  - (abs2(bi) - abs2(bi1))
        f2 = abs2(ai) - abs2(ai_1) - (abs2(bi) - abs2(bi_1))

        Fa = -1im*(-J*(sgn(f1)+sgn(f2)) * ai + g * bi)
        Fb = -1im*(J*(sgn(f1)+sgn(f2)) * bi + g * ai)
        return Fa, Fb
    end

    return [F[i][1] for i in 1:length(a)], [F[i][2] for i in 1:length(a)]
end

# Holstein-Primakoff
function F_HP(fields, params)
    a, ac = fields
    J, g, S = params

    F = map(1:length(a)) do index
        ai   = a[index]
        aci  = ac[index]
        ai1  = index+1 > length(a) ? a[1] : a[index+1]
        ai_1 = index-1 < 1 ? a[end] : a[index-1]

        aci1  = index+1 > length(ac) ? ac[1] : ac[index+1]
        aci_1 = index-1 < 1 ? ac[end] : ac[index-1]

        fact = sqrt(Complex(2*S-real(ai*aci)))

        f1 = abs(ai*aci) - abs(ai1*aci1)
        f2 = abs(ai_1*aci_1) - abs(ai*aci)

        Fa = - 1im*(-J*(sgn(f1)-sgn(f2)) * ai
            - g/2 * ai*(ai+aci)/fact + g*fact)

        FaDag = - 1im*(-J*(sgn(f1)-sgn(f2)) * aci
            - g/2 * aci*(ai+aci)/fact + g*fact)

        return Fa, FaDag
    end

    return [F[i][1] for i in 1:length(a)], [F[i][2] for i in 1:length(a)]
end

# HP + Susskind-Glogower
function F_SG(fields, params)
    n, c, s = fields
    J, g, S = params

    F = map(1:length(n)) do index
        ni = n[index]
        si = s[index]
        ci = c[index]
        ni1  = index+1 > length(n) ? n[1] : n[index+1]
        ni_1 = index-1 < 1 ? n[end] : n[index-1]

        f1 = ni - ni1
        f2 = ni - ni_1

        factor = (2*S-ni)*ni

        # Fn = -2*g*sqrt(factor)*si
        # Fc = +(-J*(sgn(f1)+sgn(f2)) + 2*g*(S-ni)*ci/sqrt(factor)) * si
        # Fs = -(-J*(sgn(f1)+sgn(f2)) + 2*g*(S-ni)*ci/sqrt(factor)) * ci

        Fn = -2*g*si
        Fc = +(-J*(sgn(f1)+sgn(f2))) * si
        Fs = -(-J*(sgn(f1)+sgn(f2))) * ci

        return Fn, Fc, Fs
    end

    return [F[i][1] for i in 1:length(n)], [F[i][2] for i in 1:length(n)], [F[i][3] for i in 1:length(n)]
end

# HP + Susskind-Glogower
function F_SG2(fields, params)
    n, phi = fields
    J, g, S = params

    F = map(1:length(n)) do index
        ni = n[index]
        phii = phi[index]
        ni1  = index+1 > length(n) ? n[1] : n[index+1]
        ni_1 = index-1 < 1 ? n[end] : n[index-1]

        f1 = ni - ni1
        f2 = ni - ni_1

        # Fn = -2*g*sqrt(factor)*si
        # Fc = +(-J*(sgn(f1)+sgn(f2)) + 2*g*(S-ni)*ci/sqrt(factor)) * si
        # Fs = -(-J*(sgn(f1)+sgn(f2)) + 2*g*(S-ni)*ci/sqrt(factor)) * ci

        Fn = -2*g*sin(phii)
        Fphi = (-2*J*(sgn(f1)+sgn(f2)))

        return Fn, Fphi
    end

    return [F[i][1] for i in 1:length(n)], [F[i][2] for i in 1:length(n)]
end

# HP + Susskind-Glogower
function F_SG3(fields, params, t)
    n   = fields[1:div(length(fields),2)]
    phi = fields[div(length(fields),2)+1:end]
    J, g, S = params

    F = map(1:length(n)) do index
        ni = n[index]
        phii = phi[index]
        ni1  = index+1 > length(n) ? n[1] : n[index+1]
        ni_1 = index-1 < 1 ? n[end] : n[index-1]

        f1 = ni - ni1
        f2 = ni - ni_1

        Fn = -2*g*sin(phii)
        Fphi = (-2*J*(sgn(f1)+sgn(f2)))

        return Fn, Fphi
    end

    return vcat([F[i][1] for i in 1:length(n)], [F[i][2] for i in 1:length(n)])
end

# Some funtion definitions

function sgn(x)
    return x >= 0 ? 1 : -1
end;

function heun_step(fields, params, F, dt)
    k1 = F(fields, params)
    k2 = F(fields .+ k1 .* dt, params)
    return fields .+ (k1 .+ k2) .* (dt/2)
end

function adaptive_heun_step(fields, params, F, dt, ϵ)
    k1 = F(fields, params)
    k2 = F(fields .+ k1 .* dt, params)

    dt_new = 0.9* dt * ϵ / findmax(abs.(vcat(k1...) .- vcat(k2...)))[1] 
    dt_new > 0.9 * dt && return dt_new, fields .+ (k1 .+ k2) .* (dt_new/2)

    return dt_new, heun_step(fields, params, F, dt_new)
end

function obs_Schw(fields, params)
    a, b = fields
    return (abs2.(a).-abs2.(b))./2
end

function obs_HP(fields, params)
    a, ac = fields
    _,_,S = params
    return S .- (a.*ac)
end

function obs_SG(fields, params)
    n,_,_ = fields
    _,_,S = params
    return S .- n
end

function obs_SG2(fields, params)
    n,_ = fields
    _,_,S = params
    return S .- n
end

function analyze_data(data, params)
    _,_,S = params
    Sz = map(data) do data_t
        return map(data_t) do d
            return d[3]
        end
    end
    signs = map(data) do data_t
        return map(data_t) do d
            return d[2]
        end
    end

    meanSz  = [sum([mean(sign[i] .* s[i]) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]
    meanSz2 = [sum([mean(sign[i] .* (s[i].^2)) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]
    absSz   = [sum([mean(sign[i] .* abs.(s[i])) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]
    # imb     = [sum([1-mean(abs.(s[i]))/S for s in Sz])/length(Sz) for i in 1:length(Sz[1])]

    # dw = [sum([sum([abs.(s1-s2) for (s1,s2) in zip(s, vcat(s[2:end],s[1]))]) for s in Sz])/length(Sz) for i in 1:length(Sz[1])]

    return DataFrame(
        t       = [real(d[1]) for d in data[1]], 
        meanSz  = meanSz,
        meanSz2 = meanSz2,
        absSz   = absSz,
        # imb     = imb,
        # dw      = dw,
        # sine    = [sum([d[i][6] for d in data])/N for i in 1:length(data[1])],
        # cosine  = [sum([d[i][7] for d in data])/N for i in 1:length(data[1])],
        # sine2   = [sum([d[i][8] for d in data])/N for i in 1:length(data[1])],
        # cosine2 = [sum([d[i][9] for d in data])/N for i in 1:length(data[1])],
        # imbS1   = [sum([d[i][10] for d in data])/N for i in 1:length(data[1])], 
    );
end

function analyze_data2(path, params)
    (g, N, num_MC, L) = params
    S = N/2

    name = path * "/TFIPerturbationTheory/data/TWA_SG_L=$(L)_Sz=$(S)_num_MC=$(num_MC)_g=$(g)"

    h5open(name*".h5", "r") do file    

        prefactor = read(file, "prefactor")[1]
        signs = read(file, "signs")
        data = map(1:num_MC) do run
            time  = read(file, "$(run)/time")
            occ   = read(file, "$(run)/occ")
            return time, occ
        end
        @show size(data[1][2])

        Sz = map(1:length(data[1][1])) do i 
            return permutedims(hcat(map(data) do data_t
                return data_t[2][i,:]
            end...))
        end
        @show size(Sz)
        @show size(Sz[1])
        

        # Sz = map(data) do data_t
        #     return map(1:size(data_t[3])[1]) do i
        #         return data_t[3][i,:]
        #     end
        # end
        # signs = [d[2] for d in data]

        meanSz   = [mean(vec(sz .* signs)) for sz in Sz]
        absSz    = [mean(vec(abs.(sz) .* signs)) for sz in Sz]
        meanSz2  = [mean(vec((sz.^2) .* signs)) for sz in Sz]
        # meanSz  = [sum([mean(sign .* s[i]) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]
        # meanSz2 = [sum([mean(sign .* (s[i].^2)) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]
        # absSz   = [sum([mean(sign .* abs.(s[i])) for (s,sign) in zip(Sz,signs)])/length(Sz) for i in 1:length(Sz[1])]

        return DataFrame(
            t       = data[1][1],
            meanSz  = prefactor * meanSz,
            meanSz2 = prefactor * meanSz2,
            absSz   = prefactor * absSz,
        );
    end
end

function analyze_data3(name, num_MC)
    # (g, N, num_MC, L) = params
    # S = N/2
    #
    # name = path * "/TFIPerturbationTheory/data/TWA_SG22_L=$(L)_Sz=$(S)_num_MC=$(num_MC)_g=$(g)"

    h5open(name*".h5", "r") do file    

        data = map(1:num_MC) do run
            time      = read(file, "$(run)/time")
            meanSz    = read(file, "$(run)/meanSz")
            absSz     = read(file, "$(run)/absSz")
            meanSz2   = read(file, "$(run)/meanSz2")
            return time, meanSz, absSz, meanSz2
        end

        meanSz   = mean([d[2] for d in data])
        absSz    = mean([d[3] for d in data])
        meanSz2  = mean([d[4] for d in data])

        return DataFrame(
            t       = data[1][1],
            meanSz  = meanSz,
            meanSz2 = meanSz2,
            absSz   = absSz,
        );
    end
end
