## diagonal part of TFI
function build_H0(spin_basis, next_neighbours, table, param)
    (L,J,g,h) = param

    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();
    
    for spin in spin_basis
        spinM = toSpinMatr(spin)
        longBorder = 0 #0.2 * (- sum(spinM[:,1]) + sum(spinM[:,end]))
        (n,D) = get(table, spin, (false,false))
        n == false && continue
        
        #xx_term
        append!(rows, [n])
        append!(columns, [n])
        append!(values, [2*D-length(next_neighbours) + longBorder])
    end
    
    return dropzeros(sparse(rows,columns,values, length(spin_basis), length(spin_basis)))
end;

## off-diagonal part of tfi
function build_H1_R1(spin_basis, table, param)
    (L,J,g,h) = param

    rowsD    = Vector{Int}(); 
    columnsD = Vector{Int}(); 
    valuesD  = Vector{Float64}();

    rowsR    = Vector{Int}(); 
    columnsR = Vector{Int}(); 
    valuesR  = Vector{Float64}();
    
    for spin in spin_basis
        (n1,D1) = get(table, spin, (false,false))
        n1 == false && continue

        for pos in 1:prod(L)
            newSpin = spinflip(spin, (pos,))
            (n2,D2) = get(table, newSpin, (false,false))
            n2 == false && continue

            if D1==D2
                append!(rowsD, [n1])
                append!(columnsD, [n2])
                append!(valuesD, [g])
            else
                append!(rowsR, [n1])
                append!(columnsR, [n2])
                append!(valuesR, [g])
            end
        end
    end
    
    H = dropzeros(sparse(rowsD,columnsD,valuesD, length(spin_basis), length(spin_basis)))
    R = dropzeros(sparse(rowsR,columnsR,valuesR, length(spin_basis), length(spin_basis)))

    return H, R
end;

## off-diagonal part of tfi
function build_V1(spin_basis, table, param)
    (L,_,g,_) = param

    rows    = Vector{Int}(); 
    columns = Vector{Int}(); 
    values  = Vector{Float64}();
    
    for spin in spin_basis
        (n1,D1) = table[spin]

        for pos in 1:prod(L)
            newSpin = spinflip(spin, (pos,))
            (n2,D2) = table[newSpin]

            append!(rows, [n1])
            append!(columns, [n2])
            append!(values, [g])
        end
    end
    
    return dropzeros(sparse(rows,columns,values, length(spin_basis), length(spin_basis)))
end;

function build_V1_mod(H0, V, nRef, E)
    rows = Vector{Int}(); 
    columns = Vector{Int}(); 
    values = Vector{Float64}();

    for (r,c,v) in zip(findnz(V)...)
        if r != nRef
            append!(rows, [r])
            append!(columns, [c])
            append!(values, [v/(E-H0[r,r])])
        end
    end
    
    return dropzeros(sparse(rows,columns,values, size(V)[1], size(V)[1]))
end;

