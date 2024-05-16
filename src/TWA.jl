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

        Fn = -2*g*sqrt(factor)*si
        Fc = +(-J*(sgn(f1)+sgn(f2)) + 2*g*(S-ni)*ci/sqrt(factor)) * si
        Fs = -(-J*(sgn(f1)+sgn(f2)) + 2*g*(S-ni)*ci/sqrt(factor)) * ci

        return Fn, Fc, Fs
    end

    return [F[i][1] for i in 1:length(n)], [F[i][2] for i in 1:length(n)], [F[i][3] for i in 1:length(n)]
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

function analyze_data(data, params)
    _,_,S = params
    Sz = map(data) do data_t
        return map(data_t) do d
            return d[2:end]
        end
    end

    meanSz = [sum([mean(s[i]) for s in Sz])/length(Sz) for i in 1:length(Sz[1])]
    # absSz  = [sum([mean(abs.(s[i])) for s in Sz])/length(Sz) for i in 1:length(Sz[1])]
    imb    = [sum([1-mean(abs.(s[i]))/S for s in Sz])/length(Sz) for i in 1:length(Sz[1])]

    # dw = [sum([sum([abs.(s1-s2) for (s1,s2) in zip(s, vcat(s[2:end],s[1]))]) for s in Sz])/length(Sz) for i in 1:length(Sz[1])]

    return DataFrame(
        t      = [real(d[1]) for d in data[1]], 
        meanSz = meanSz,
        imb    = imb,
        # dw      = dw,
        # sine    = [sum([d[i][6] for d in data])/N for i in 1:length(data[1])],
        # cosine  = [sum([d[i][7] for d in data])/N for i in 1:length(data[1])],
        # sine2   = [sum([d[i][8] for d in data])/N for i in 1:length(data[1])],
        # cosine2 = [sum([d[i][9] for d in data])/N for i in 1:length(data[1])],
        # imbS1   = [sum([d[i][10] for d in data])/N for i in 1:length(data[1])], 
    );
end
