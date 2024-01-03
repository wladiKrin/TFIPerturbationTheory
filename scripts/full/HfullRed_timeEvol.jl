include("../src/SW.jl")

gs = [-0.5,-0.7,-1.0,-1.25,-1.5,-1.75,-2.0,-2.5]
# ns = [1,2,3,4,5,6,7]
# params = collect(Iterators.product(ns,gs))

#for (n_order,g) in zip(ns,gs)
#let
for g in gs
  L= (4,4)
  N = prod(L);
  J = -1;
  h = -0.

  ts = [0.0,0.1]
  step = 1.1
  tmax = 1e20

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

  H0  = build_H0(spin_basis, next_neighbours, spin_basis_table, (L,J,g,h));
  imb_precalc = map(s -> imbalance(s, L), spin_basis)
  twoCorrelM  = build_twoCorrel(spin_basis, spin_basis_table)
  xPolM       = build_xPol(spin_basis, spin_basis_table)

  @show g
  vals,vecs,dw = readSpec("../data/spec_ED_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h)")

  is10 = [i for (i,dw) in enumerate(dw) if dw > 9 && dw<11]
  is8   = [i for (i,dw) in enumerate(dw) if dw > 7 && dw<9]
  is = vcat(is8,is10)
  println("built is")


  ### initial state ###
  init_idx = 476

  psi=zeros(length(spin_basis))
  psi[init_idx]=1
  psi = sparse(psi)


  ### map initial state onto eigenbasis of H ###
  vecsRed = map(is) do i
    return vecs[:,i]
  end
  valsRed = map(is) do i
    return vals[i]
  end

  vecsRed = Transpose(permutedims(hcat(vecsRed...)))

  psi = Transpose(vecsRed) * psi

  data = Any[]

  for (t, tf) in zip(ts[1:end-1], ts[2:end])
    dt = tf-t

    @time "time t= $(round(t, digits=1))" begin
      psi_prime = vecsRed*psi

      imb = mapreduce(+, enumerate(psi_prime)) do (i,psi_i)
        α = abs2(psi_i)
        return α * imb_precalc[i]
      end

      N      = real.(dot(psi_prime, H0      * psi_prime))
      xPol   = real.(dot(psi_prime, xPolM   * psi_prime))
      xCorr  = real.(dot(psi_prime, twoCorrelM      * psi_prime))

      append!(data, [[t, imb, N, xPol, xCorr]])

      # Propagate state
      U = exp.(-1im*dt .* valsRed)
      psi = U .* psi
    end

    #vecs = sort([(i,p) for (i,p) in enumerate(abs2.(psi_prime)) if p>1e-3], by=x->x[2], rev=true)
    #@show [v[1] for v in vecs]
  end

  df = DataFrame(t = [real(d[1]) for d in data], imb = [real(d[2]) for d in data], N = [real(d[3]) for d in data], xPol = [real(d[4]) for d in data], xCorr = [real(d[5]) for d in data])
  CSV.write("../data/timeEv_dw8_10states_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)
  #CSV.write("../data/obs_EDred_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", df)

  #dfwf = DataFrame(psi = psi)
  #CSV.write("../data/wf_ED_longtime_L=($(L[1])_$(L[2]))_J=$(J)_g=$(g)_h=$(h).csv", dfwf);
end
