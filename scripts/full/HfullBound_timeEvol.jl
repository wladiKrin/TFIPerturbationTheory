include("../../src/PertTheory.jl")

# gs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.75,-0.8,-0.85,-0.9,-0.95,-1.0,-1.25,-1.5,-1.75,-2.0]
gs = [-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0]

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

function corrF(spins::Tuple{Vararg{Int64}}, L)
    spins = reshape([s for s in spins], L)

    left = mapreduce(+, Iterators.product(1:L[1], 1:floor(Int, L[2]/2))) do (i,j)
        spin = (spins[i,j] == 1) ? 1 : -1
        return spin
    end/(prod(L)/2)
    right = mapreduce(+, Iterators.product(1:L[1], floor(Int, L[2]/2)+1:L[2])) do (i,j)
        spin = (spins[i,j] == 1) ? 1 : -1
        return spin
    end/(prod(L)/2)
    return left*right
end

let
  L= (4,4)
  N = prod(L);
  J = -1;
  h = -0.

  array_id = parse(Int, ARGS[1])
  g = gs[array_id]

  ts = [0.0,0.1]
  step = 1.01
  tmax = 1e10

  ### logarithmic timesteps ###
  while true
    push!(ts, ts[end]*step)
    ts[end] > tmax && break
  end

  next_neighbours = nearest_neighbours(L, collect(1:prod(L)), periodic_y=false)

  spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
  dw_precalc = map(spin_basis) do spin
      return domainWallL(spin, L, next_neighbours)
  end

  ### sort basis according to domain wall length ###
  sorted_spin_basis = sort(collect(zip(dw_precalc, spin_basis)), by = x->x[1])
  dw_precalc  = [d[1] for d in sorted_spin_basis]
  spin_basis  = [d[2] for d in sorted_spin_basis]

  spin_basis_table = Dict(
      map(enumerate(zip(spin_basis,dw_precalc))) do (i, (spin, dw))
          return (spin, (i,dw))
      end
  );

  domainWall_table = Dict(
      map(enumerate(dw_precalc)) do (i,  dw)
          return (i, dw)
      end
  );

  ### initial state ###
  init_spin = vcat(fill(1,Int(N/2)),fill(0,Int(N/2)))
  init_idx = first(spin_basis_table[Tuple(init_spin)]);

  psi1=zeros(length(spin_basis))
  psi1[init_idx]=1
  psi1 = sparse(psi1)

  #=
  init_spinC = vcat(fill(0,Int(N/2)),fill(1,Int(N/2)))
  init_idxC = first(spin_basis_table[Tuple(init_spinC)]);

  psiC=zeros(length(spin_basis))
  psiC[init_idxC]=1
  psiC = sparse(psiC)

  (p, phi) = (0.9, pi/2)
  =#

  psi = psi1 # sqrt(p)*psi1 + sqrt(1-p)*exp(1im*phi)*psiC

  ### build Hamiltonians ###
  H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
  H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
  V1 = H1+R1;

  H_full = H0+V1

  ### observables ###
  imb_precalc = map(s -> imbalance(s, L), spin_basis)
  imbS1_precalc = map(s -> imbalanceStrip(s, L, 1), spin_basis)
  corr_precalc = map(s -> corrF(s, L), spin_basis)
  twoCorrelM  = build_twoCorrel(spin_basis, spin_basis_table)
  xPolM       = build_xPol(spin_basis, spin_basis_table)
  #D0          = dropzeros(reduce(+, vecH))
  #H0_corr     = build_corr(H0, vecS)
  #D0_corr     = build_corr(D0, vecS)

  vals,vecs,dw = readSpec("../../data/spec_ED_Bound_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

  ### map initial state onto eigenbasis of H ###
  psi = Transpose(vecs) * psi

  data = Any[]

  for (t, tf) in zip(ts[1:end-1], ts[2:end])
    dt = tf-t

    @time "time t= $(round(t, digits=1))" begin
      psi_prime = vecs*psi

      imb = mapreduce(+, enumerate(psi_prime)) do (i,psi_i)
        α = abs2(psi_i)
        return α * imb_precalc[i]
      end

      imbS1 = mapreduce(+, enumerate(psi_prime)) do (i,psi_i)
        alpha = abs2(psi_i)
        return alpha * imbS1_precalc[i]
      end

      corr = mapreduce(+, enumerate(psi_prime)) do (i,psi_i)
        alpha = abs2(psi_i)
        return alpha * corr_precalc[i]
      end

      N         = real.(dot(psi_prime, H0      * psi_prime))

      twoCorrel = dot(psi_prime, twoCorrelM*psi_prime)
      xPol      = dot(psi_prime, xPolM*psi_prime)

      append!(data, [[t, imb, N, xPol, twoCorrel, corr, imbS1]])

      # Propagate state
      U = exp.(-1im*dt .* vals)
      psi = U .* psi
    end
  end

  df = DataFrame(
    t = [real(d[1]) for d in data], 
    imb = [real(d[2]) for d in data], 
    N = [real(d[3]) for d in data], 
    xPol = [real(d[4]) for d in data], 
    twoCorrel = [real(d[5]) for d in data],
    corr = [real(d[6]) for d in data],
    imbS1 = [real(d[7]) for d in data],
  )
  CSV.write("../../data/obs_ED_Bound_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
end
