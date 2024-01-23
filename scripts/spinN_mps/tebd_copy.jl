using ITensors
using CSV, DataFrames

let
  N = 10
  cutoff = 1E-15
  tau = 0.001
  ttotal = 0.1
  (J,g) = (0,-1.)

  # Make an array of 'site' indices

  # Ising model ##
  s = siteinds("S=1/2", N)
  gates = ITensor[]
  for j in 1:(N - 1)
    s1 = s[j]
    s2 = s[j + 1]
    hj = J*op("Z", s1) * op("Z", s2)
    Gj = exp(-im * tau / 2 * hj)
    push!(gates, Gj)
  end
  
  for j in 1:(N)
    s1 = s[j]
    hj = g*op("X", s1)
    Gj = exp(-im * tau / 2 * hj)
    push!(gates, Gj)
  end
  
  ## rung decoupled heisenberg chain model ##
  # gates = ITensor[]
  # for j in 1:(N - 2)
  #   s1 = s[j]
  #   s2 = s[j + 2]
  #   J = iseven(j) ? J1 : J2
  #
  #   hj = 2*J*op("S+", s1)*op("S-", s2) +
  #        2*J*op("S-", s1)*op("S+", s2) +
  #        J*op("Sz", s1)*op("Sz", s2)
  #
  #   Gj = exp(-im * tau / 2 * hj)
  #   push!(gates, Gj)
  # end
  #
  # for j in 1:(N)
  #   s1 = s[j]
  #   hj = g*op("Sz", s1)
  #   Gj = exp(-im * tau / 2 * hj)
  #   push!(gates, Gj)
  # end

  # Include gates in reverse order too
  # (N,N-1),(N-1,N-2),...
  append!(gates, reverse(gates))
  @show typeof(gates)

  # psi = MPS(s, n -> isodd(n) ? "Up" : "Dn")
  # psi = MPS(s, "Up")
  psi = MPS(s, x->["Up","Up","Up","Dn","Up","Dn","Up","Dn","Up","Up"][x])
  @show maxlinkdim(psi)

  data = []

  for t in 0.0:tau:ttotal+tau
    # println("time t = ", round(t, digits = 1))
    t≈ttotal && break

    expec = real(expect(psi, "Sz", sites=5))
    Sz = sum(expec)/length(expec)
    psi = apply(gates, psi; cutoff)
    normalize!(psi)
    @show Sz
    # Sz = sum(real((expect(psi, "Z"))))/N
    # Sx = sum(real((expect(psi, "X"))))/N
    push!(data, [t, Sz])

  end

  # df = DataFrame(time = [d[1] for d in data], zPol = [d[2] for d in data]) #, xPol = [d[3] for d in data])
  # name_obs = "data/heisenbergqns_tebd_N=$(N)"
  # CSV.write(name_obs*".csv", df)

  @show maxlinkdim(psi)
end
