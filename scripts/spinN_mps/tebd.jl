using ITensors
using CSV, DataFrames
using LinearAlgebra

let
  N = 4
  D = 5
  cutoff = 0 #1E-14
  tau = 0.001
  ttotal = 0.1
  (J,g) = (-1.,-1.5)

  function ITensors.op(::OpName"Nabs", ::SiteType"Qudit", s1::Index, s2::Index)
    inds = vcat(collect(Iterators.product(1:dim(s1), 1:dim(s2)))...)
    mat = map(Iterators.product(inds,inds)) do ((iIn,jIn),(iOut,jOut))
      if iIn == iOut && jIn == jOut
        return (2*length(s1)-3) - 2 * abs(iIn-jIn)
      end
      return 0
    end
    return itensor(mat, s2', s1', s2, s1)
  end

  function ITensors.op(::OpName"N2", ::SiteType"Qudit", s::Index)
    # @show op("N", s) * prime(op("N", s))
    # return op("N", s) * op("N", s)
    mat = diagm(collect(0:dim(s)-1).^2)
    return itensor(mat, s', s)
  end

  ## Ising model ##
  s = siteinds("Qudit", N; dim = 8)

  gates = ITensor[]
  for (j1,j2) in zip(1:N, vcat(2:N,1))
    s1 = s[j1]
    s2 = s[j2]

    hj = J*op("Nabs", s1, s2)
    Gj = exp(-im * tau / 2 * hj)

    push!(gates, Gj)
  end
  
  for j in 1:N
    s1 = s[j]
    hj = g*(op("Adag", s1) + op("A", s1))
    Gj = exp(-im * tau / 2 * hj)
    push!(gates, Gj)
  end
  
  # Include gates in reverse order too
  # (N,N-1),(N-1,N-2),...
  append!(gates, reverse(gates))

  # psi = MPS(s, x->["1","2","3","4","4","4","4","4"][x])

  psi = MPS(s, "$(div(D-1,2))")
  data = []

  for t in 0.0:tau:ttotal+tau
    println("time t = ", round(t, digits = 1))
    t≈ttotal && break
    s1 = s[1]
    # @show maxlinkdim(exp(-im * tau / 2 * g*(op("Adag", s1) + op("A", s1))))
    # @show apply(exp(-im * tau / 2 * g*(op("Adag", s1) + op("A", s1))), psi; maxlinkdim=10)

    meanN = expect(psi, "N") 
    varN = expect(psi, "N2") .- meanN.^2
    @show meanN
    @show varN

    psi = apply(gates, psi; cutoff)

    normalize!(psi)
    push!(data, [t, meanN])

  end
  @show psi

  # df = DataFrame(time = [d[1] for d in data], zPol = [d[2] for d in data]) #, xPol = [d[3] for d in data])
  # name_obs = "data/heisenbergqns_tebd_N=$(N)"
  # CSV.write(name_obs*".csv", df)
end
