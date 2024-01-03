## energies ##

function BW_energy(nState, H0, V)
    f(x) = x - (H0[nState,nState] + V[nState,nState] + BW_second_order_energy(nState, H0, V, x) + BW_third_order_energy(nState, H0, V, x))
    @time res = find_zero(f, 0)
    return res
end

function BW_zero_order_energy(nState, H0, V, E)
    return H0[nState,nState]
end

function BW_first_order_energy(nState, H0, V, E)
    return V[nState,nState]
end

function BW_second_order_energy(nState, H0, V, E)
    inds, vals = findnz(V[nState,:])
    return mapreduce(+, zip(inds, vals)) do (i,v)
        nState == i && return 0
        DE = E-H0[i,i]
        return abs2(v)/DE
    end
end

function BW_third_order_energy(nState, H0, V, E)
    inds1, vals1 = findnz(V[nState,:])
    return mapreduce(+, zip(inds1, vals1)) do (k1,v1)
        nState == k1 && return 0
        DE1 = E - H0[k1,k1]

        inds2, vals2 = findnz(V[k1,:])
        return mapreduce(+, zip(inds2, vals2)) do (k2,v2)
            nState == k2 && return 0
            DE2 = E - H0[k2,k2]


            return v2*v1*V[k2,nState]/(DE1*DE2)
        end
    end
end

function BW_fourth_order_energy(nState, H0, V, E)
    inds1, vals1 = findnz(V[nState,:])
    return mapreduce(+, zip(inds1, vals1)) do (k1,v1)
        nState == k1 && return 0
        DE1 = E - H0[k1,k1]

        inds2, vals2 = findnz(V[k1,:])
        return mapreduce(+, zip(inds2, vals2)) do (k2,v2)
            nState == k2 && return 0
            DE2 = E - H0[k2,k2]

            inds3, vals3 = findnz(V[k2,:])
            return mapreduce(+, zip(inds3, vals3)) do (k3,v3)
                nState == k3 && return 0
                DE3 = E - H0[k3,k3]

                return v3*v2*v1*V[nState,k3]/(DE1*DE2*DE3)
            end
        end
    end
end

function BW_sixth_order_energy(nState, H0, V, E)
    inds1, vals1 = findnz(V[nState,:])
    return mapreduce(+, zip(inds1, vals1)) do (k1,v1)
        nState == k1 && return 0
        DE1 = E - H0[k1,k1]

        inds2, vals2 = findnz(V[k1,:])
        return mapreduce(+, zip(inds2, vals2)) do (k2,v2)
            nState == k2 && return 0
            DE2 = E - H0[k2,k2]

            inds3, vals3 = findnz(V[k2,:])
            return mapreduce(+, zip(inds3, vals3)) do (k3,v3)
                nState == k3 && return 0
                DE3 = E - H0[k3,k3]

                inds4, vals4 = findnz(V[k3,:])
                return mapreduce(+, zip(inds4, vals4)) do (k4,v4)
                    nState == k4 && return 0
                    DE4 = E - H0[k4,k4]

                    inds5, vals5 = findnz(V[k4,:])
                    return mapreduce(+, zip(inds5, vals5)) do (k5,v5)
                        nState == k5 && return 0
                        DE5 = E - H0[k5,k5]
                        return v5*v4*v3*v2*v1*V[k5,nState]/(DE1*DE2*DE3*DE4*DE5)
                    end
                end
            end
        end
    end
end

function BW_eighth_order_energy(nState, H0, V, E)
    inds1, vals1 = findnz(V[nState,:])
    return mapreduce(+, zip(inds1, vals1)) do (k1,v1)
        nState == k1 && return 0
        DE1 = E - H0[k1,k1]

        inds2, vals2 = findnz(V[k1,:])
        return mapreduce(+, zip(inds2, vals2)) do (k2,v2)
            nState == k2 && return 0
            DE2 = E - H0[k2,k2]

            inds3, vals3 = findnz(V[k2,:])
            return mapreduce(+, zip(inds3, vals3)) do (k3,v3)
                nState == k3 && return 0
                DE3 = E - H0[k3,k3]

                inds4, vals4 = findnz(V[k3,:])
                return mapreduce(+, zip(inds4, vals4)) do (k4,v4)
                    nState == k4 && return 0
                    DE4 = E - H0[k4,k4]

                    inds5, vals5 = findnz(V[k4,:])
                    return mapreduce(+, zip(inds5, vals5)) do (k5,v5)
                        nState == k5 && return 0
                        DE5 = E - H0[k5,k5]

                        inds6, vals6 = findnz(V[k5,:])
                        return mapreduce(+, zip(inds6, vals6)) do (k6,v6)
                            nState == k6 && return 0
                            DE6 = E - H0[k6,k6]

                            inds7, vals7 = findnz(V[k6,:])
                            return mapreduce(+, zip(inds7, vals7)) do (k7,v7)
                                nState == k7 && return 0
                                DE7 = E - H0[k7,k7]
                                return v7*v6*v5*v4*v3*v2*v1*V[k7,nState]/(DE1*DE2*DE3*DE4*DE5*DE6*DE7)
                            end
                        end
                    end
                end
            end
        end
    end
end

function BW_eighth_order_energyMod(nState, H0, V, E)
    Vmod = build_V1_mod(H0, V, nState, E);
    VmodVmod = sparse(Vmod*Vmod)
    VVmod = sparse(V*Vmod)
    fourthOrd = VVmod*VmodVmod
    sixthOrd = fourthOrd*VmodVmod
    eighthOrd = sixthOrd*VmodVmod
    return (VVmod)[nState,nState] + (fourthOrd)[nState,nState] + (sixthOrd)[nState,nState] + (eighthOrd)[nState,nState]
end


function BW_nth_order_energy(maxOrder, nState, H0, V, E)

    maxOrder ≤ 1 && return H0[nState,nState]
    indsStart, valsStart = findnz(V[nState,:])

    function iter(inds, vals, order, maxOrder)
        if order == maxOrder
            return mapreduce(+, zip(inds, vals)) do (k,v)
                nState == k && return 0
                DE = E - H0[k,k]
                return v*V[k,nState]/DE
            end
        else
            res = 0
            for (k,v) in zip(inds, vals)
                nState == k && continue
                DE = E - H0[k,k]

                indsNext, valsNext = findnz(V[k,:])
                iterResult = iter(indsNext, valsNext, order+1, maxOrder)

                if order%2 == 0
                    res += iterResult*v*V[k,nState]/DE
                end
                res += iterResult*v/DE
            end
            return res
        end
    end
    
    return iter(indsStart, valsStart, 2, maxOrder)
end

## states ##

function BW_zeroth_order_states(nState, H0, V, E)
    return sparsevec([nState], [1], 2^16)
end

function BW_first_order_states(nState, H0, V, E)
    inds1, vals1 = findnz(V[nState,:])

    vals = map(zip(inds1, vals1)) do (k1,v1)
        nState == k1 && return 0
        DE1 = E - H0[k1,k1]
        return v1/DE1
    end

    return sparsevec(inds1, vals, length(basis_states))
end

function first_order_states(nState, basis_states, H0, V)
    inds2, vals2 = findnz(V[nState,:])
    vals = map(zip(inds2, vals2)) do (k2,v2)
        nState == k2 && return 0
        DE2 = H0[nState,nState] - H0[k2,k2]
        return v2/DE2
    end
    return SparseVector(length(basis_states), inds2, vals)
end

function BW_second_order_states(nState, H0, V, E)
    inds1, vals1 = findnz(V[nState,:])

    res_inds = []
    res_vals = mapreduce(+, zip(inds1, vals1)) do (k1,v1)
        nState == k1 && return 0
        DE1 = E - H0[k1,k1]

        inds2, vals2 = findnz(V[k1,:])
        return map(zip(inds2, vals2)) do (k2,v2)
            nState == k2 && return 0
            DE2 = E - H0[k2,k2]

            push!(inds, k2)

            return v1*v2/(DE1*DE2)
        end
    end


    return SparseVector(length(basis_states), res_inds, res_vals)
end

## not working ##
#=
function BW_eighth_order_energyModGPU(nState, H0, V, E)
    Vmod = cu(Matrix(build_V1_mod(H0, V, nState, E)))
    Vmod2 = Vmod*Vmod
    resM = cu(Matrix(V))*Vmod
    CUDA.unsafe_free!(Vmod)
    CUDA.reclaim()
    res = Array(resM)[nState,nState] # second
    resM = resM*Vmod2
    res += Array(resM)[nState,nState] # fourth
    resM = resM*Vmod2
    res += Array(resM)[nState,nState] # sixth
    resM = resM*Vmod2
    res += Array(resM)[nState,nState] # eighth
    return res
end

function BW_eighth_order_energyModGPU(nState, H0, V, E)
    CUDA.memory_status()
#   Vmod = cu(Matrix(build_V1_mod(H0, V, nState, E)))
#   Vmod2 = Vmod*Vmod
#   resM = cu(Matrix(V))*Vmod
#   CUDA.unsafe_free!(Vmod)
#   CUDA.reclaim()
#   CUDA.memory_status()
#   res = Array(resM)[nState,nState] # second
#   resM = resM*Vmod2
#   CUDA.reclaim()
#   CUDA.memory_status()
#   res += Array(resM)[nState,nState] # fourth
#   resM = resM*Vmod2
#   CUDA.reclaim()
#   CUDA.memory_status()
#   res += Array(resM)[nState,nState] # sixth
#   resM = resM*Vmod2
#   CUDA.reclaim()
#   CUDA.memory_status()
#   res += Array(resM)[nState,nState] # eighth
#   CUDA.reclaim()
#   CUDA.memory_status()
#   return res

    Vmod = CuSparseMatrixCSR(cu(Matrix(build_V1_mod(H0, V, nState, E))))
    VmodT = CUSPARSE.gemm('N', 'N', 1,  Vmod, Vmod, 'O')
    # VmodT = CUSPARSE.geam(1, Vmod, 1, Vmod, 'O', 'O', 'O')
    # VVmod = cu(Matrix(V))*Vmod 


    res = CUSPARSE.gemm('N', 'N', 1,  Vmod, Vmod, 'O')
    CUDA.unsafe_free!(Vmod)
    CUDA.reclaim()
    res = CUSPARSE.gemm('N', 'N', 1,  res, VmodT, 'O')
    CUDA.memory_status()
    res = CUSPARSE.gemm('N', 'N', 1,  res, VmodT, 'O')
    CUDA.memory_status()
    CUDA.unsafe_free!(VmodT)
    CUDA.reclaim()
    CUDA.memory_status()
    #res = VVmod + VVmod*res 
    return Array(res)[nState,nState]
end
=#
