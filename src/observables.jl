## matrix representation of spin configuration
function toSpinMatr(spins::Tuple{Vararg{Int64}}, L)
    return reshape([s for s in spins], L)
end

## imbalance
function imbalance(spins::Tuple{Vararg{Int64}}, L)
    spins = reshape([s for s in spins], L)

    return mapreduce(+, Iterators.product(1:L[1], 1:L[2])) do (i,j)
        if j <= L[1]/2 
            spin = (spins[i,j] == 1) ? 1 : -1
        else
            spin = (spins[i,j] == 1) ? -1 : 1
        end
        return spin
    end/prod(L)
end

## domain wall length operator;;; not very efficient just use H0 for that
function domainWallL(spins::Tuple{Vararg{Int64}}, L, neigh)
    spins = reshape([s for s in spins], L)

    D = 0
    for (i,j) in neigh
        pos1 = _coordinate_simple_lattice(i, L)
        pos2 = _coordinate_simple_lattice(j, L)
        s1 = spins[pos1[1],pos1[2]] == 1 ? 1 : -1
        s2 = spins[pos2[1],pos2[2]] == 1 ? 1 : -1
        D += (1-s1*s2)/2
    end
    return trunc(Int,D)
end;

## build correction for the quasi-conserved dressed operator Op (N & D)
function build_corr(Op, vecS)
  res = map(1:length(vecS)) do n
    part = makePartition(n)

    return dropzeros(mapreduce(+, part) do p
      SMatr = map(i->-vecS[i], p)
      return 1/factorial(length(p)) * reduce(commutator, SMatr[2:end]; init=commutator(SMatr[1], Op))
    end)
  end

  primePart = makePrimePartition(length(vecS)+1)

  push!(res, mapreduce(+, primePart) do p
    SMatr = map(i->-vecS[i], p)
    return 1/factorial(length(p)) * reduce(commutator, SMatr[2:end], init=commutator(SMatr[1], Op))
  end)

  return dropzeros(reduce(+, res))
end;

## x-polarization
function build_xPol(spin_basis, table)
    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();
    
    for spin in spin_basis
        (n,_) = table[spin]

        D = mapreduce(+, spin) do s
            return 2*s-1
        end / length(spin)
        
        #xx_term
        append!(rows, [n])
        append!(columns, [n])
        append!(values, [D])
    end
    
    return dropzeros(sparse(rows,columns,values))
end;

function build_twoCorrel(spin_basis, table)
    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();
    
    for spin in spin_basis
        (n,_) = table[spin]

        D = mapreduce(+, Iterators.product(spin,spin)) do (s1,s2)
            return (2*s1-1)*(2*s2-1)
        end / length(spin)^2
        
        #xx_term
        append!(rows, [n])
        append!(columns, [n])
        append!(values, [D])
    end
    
    return dropzeros(sparse(rows,columns,values))
end;
