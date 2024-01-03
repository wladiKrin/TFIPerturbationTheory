include("../src/SW.jl")

gs = [-0.1,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0,-1.25,-1.5,-1.75,-2.0,-2.25,-2.5]

for g in gs
  L= (4,4)
  N = prod(L);
  J = -1;
  h = -0.

  #g = gs[parse(Int, ARGS[1])]
  #g = -0.5

  n_order = 1

  ts = [0.0,0.1]
  step = 1.1
  tmax = 1e50

  ### logarithmic timesteps ###
  while true
    push!(ts, ts[end]*step)
    ts[end] > tmax && break
  end

  next_neighbours = nearest_neighbours(L, collect(1:prod(L)))

  spin_basis = vec(collect(Iterators.product(fill([1,0],N)...)));
  dw_precalc = map(spin_basis) do spin
      return domainWallL(spin, L)
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

  psi=zeros(length(spin_basis))
  psi[init_idx]=1
  psi = sparse(psi)

  ### build Hamiltonians ###
  H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
  H1, R1 = build_H1_R1(spin_basis, spin_basis_table, (L,J,g,h));
  V1 = H1+R1;

  #======================== full =============================#
  H_full = H0+V1

  #==================== perturbative =========================#
  vecH, vecR, vecS = SW_transformation(H0, H1, R1, domainWall_table, n_order)
  H_pert = dropzeros(reduce(+, vecH; init=H0))

  ### observables ###
  imb_precalc = map(s -> imbalance(s, L), spin_basis)
  #twoCorrelM  = build_twoCorrel(spin_basis, spin_basis_table)
  #xPolM       = build_xPol(spin_basis, spin_basis_table)
  #D0          = dropzeros(reduce(+, vecH))
  #H0_corr     = build_corr(H0, vecS)
  #D0_corr     = build_corr(D0, vecS)
  N = sparse(diagm(fill(16,2^16)) + H0 ./2)

  vals,vecs,dw = readSpec("../data/spec_ED_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

  ### map initial state onto eigenbasis of H ###
  psi = Transpose(vecs) * psi
  @show g, dot(psi,psi)

  df = DataFrame(occRe = real.(vecs[476,:]), occIm = imag.(vecs[476,:]), dw = real.(dw), en = real.(vals))
  CSV.write("../data/occTest_ED_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
  continue

  data = sort(enumerate(dw), by=x->x[2], rev= true)
  is8 = [i for (i,dw) in data if dw > 7 && dw<9]
  is10 = [i for (i,dw) in data if dw > 9 && dw<11]
  is10 = vcat(is10, is8)
  #=
  obs = mapreduce(.+, Iterators.product(is,is)) do (i,j)
      imb = conj(vecs[init_idx,i])*vecs[init_idx,j]*mapreduce(+, enumerate(zip(vecs[:,i],vecs[:,j]))) do (num,(v1,v2))
          return conj(v1) * v2 * imb_precalc[num]
      end
      dw = conj(vecs[init_idx,i])*vecs[init_idx,j]*dot(vecs[:,i],N*vecs[:,j])
      return [imb, dw]
  end
  occSum = sum(map(x->abs2(vecs[476,x]), is))
  @show obs, occSum
  =#

  dwNew = [dw[i] for (i,v) in data]
  en = [vals[i] for (i,v) in data]


  data = Any[]

  for (t, tf) in zip(ts[1:end-1], ts[2:end])
    dt = tf-t

    @time "time t= $(round(t, digits=1))" begin
      psi_prime = vecs*psi

      #=
      imb = mapreduce(+, enumerate(psi_prime)) do (i,psi_i)
        α = abs2(psi_i)
        return α * imb_precalc[i]
      end

      N         = real.(dot(psi_prime, H0      * psi_prime))
      N_corr    = real.(dot(psi_prime, H0_corr * psi_prime))

      D         = dot(psi_prime, D0*psi_prime)
      D_corr    = 0# dot(psi_prime, D0_corr*psi_prime)
      @show 16 + N/2 
      @show real.(dot(psi_prime, Hnew * psi_prime))


      twoCorrel = dot(psi_prime, twoCorrelM*psi_prime)
      xPol      = dot(psi_prime, xPolM*psi_prime)

      append!(data, [[t, imb, N, N_corr, D, D_corr, xPol, twoCorrel]])
      =#

      # Propagate state
      U = exp.(-1im*dt .* vals)
      psi = U .* psi
    end

    vs = sort([(i,p) for (i,p) in enumerate(abs2.(psi_prime)) if p>1e-3], by=x->x[2], rev=true)
    @show [v[1] for v in vs]
  end
  return

  df = DataFrame(t = [real(d[1]) for d in data], imb = [real(d[2]) for d in data], N = [real(d[3]) for d in data], N_corr = [real(d[4]) for d in data], D = [real(d[5]) for d in data], D_corr = [real(d[6]) for d in data], xPol = [real(d[7]) for d in data], twoCorrel = [real(d[8]) for d in data])
  CSV.write("../data/obs_ED_n=$(n_order)_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
end
