function makePartition(n::Int)
    part = vcat(collect(Combinatorics.partitions(n)))
    return vcat(map(part) do p
        return rmDoubles(collect(Combinatorics.permutations(p)))
    end...)
end

function makePrimePartition(n::Int)
    return filter(x->length(x)>1, makePartition(n))
end

function commutator(A,B)
    return A*B-B*A
end

function build_S(table, R)
    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();

    for (r,c,v) in zip(findnz(R)...)
        D1 = table[r]
        D2 = table[c]

        append!(rows, [r])
        append!(columns, [c])
        append!(values, [v/(2*(D1-D2))])
    end
    
    return dropzeros(sparse(rows,columns,values, 2^16, 2^16))
end;

function build_V(H0, V, vecS)
    n = length(vecS)
    part      = makePartition(n)
    primePart = makePrimePartition(n+1)

    res1 = mapreduce(+, part) do p
        SMatr = map(p) do i 
            return vecS[i]
        end
        return 1/factorial(length(p)) * reduce(commutator, SMatr[2:end]; init=commutator(SMatr[1], V))
    end

    res2 = mapreduce(+, primePart) do p
        SMatr = map(p) do i 
            return vecS[i]
        end
        return 1/factorial(length(p)) * reduce(commutator, SMatr[2:end]; init=commutator(SMatr[1], H0))
    end

    return dropzeros(res1 + res2)
end;

function build_H_R(table, V)
    rowsD    = Vector{Int}(); 
    columnsD = Vector{Int}(); 
    valuesD  = Vector{Float64}();

    rowsR    = Vector{Int}(); 
    columnsR = Vector{Int}(); 
    valuesR  = Vector{Float64}();

    for (r,c,v) in zip(findnz(V)...)
        D1 = table[r]
        D2 = table[c]

        if D1==D2
            append!(rowsD, [r])
            append!(columnsD, [c])
            append!(valuesD, [v])
        else
            append!(rowsR, [r])
            append!(columnsR, [c])
            append!(valuesR, [v])
        end
    end
    
    H = dropzeros(sparse(rowsD,columnsD,valuesD, 2^16, 2^16))
    R = dropzeros(sparse(rowsR,columnsR,valuesR, 2^16, 2^16))

    return H, R
end;


function SW_transformation(H0, H1, R1, tableDW, n_order)
  vecH = [H1]
  vecR = [R1]
  vecS = []

  V1 = H1+R1

  for i in 1:n_order
      println("current SW-order = $(i)")
      @time "Calculating S:" S = build_S(tableDW, vecR[end]);
      push!(vecS, S)

      test = dropzeros(commutator(S,H0))+vecR[end]
      if nnz(droptol!(test, 1e-12)) != 0
        @show findnz(droptol!(test, 1e-12))
        @error(" [S_$(i), H_0] + R_$(i) =/= 0 ! ")
      end

      @time "Calculating V:" V = build_V(H0, V1, vecS);
      @time "Calculating D, R" H, R = build_H_R(tableDW, V)
      push!(vecH, H)
      push!(vecR, R)
  end

  return vecH, vecR, vecS
end
