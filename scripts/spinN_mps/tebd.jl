using ITensors
using CSV, DataFrames
using LinearAlgebra

function ITensors.op(::OpName"Nabs", ::SiteType"Qudit", s1::Index, s2::Index)
  inds = vcat(collect(Iterators.product(1:dim(s1), 1:dim(s2)))...)
  mat = map(Iterators.product(inds,inds)) do ((iIn,jIn),(iOut,jOut))
    if iIn == iOut && jIn == jOut
      return (2*(dim(s1)-1)-3) - 2 * abs(iIn-jIn)
    end
    return 0
  end
  return itensor(mat, s2', s1', s2, s1)
end

function ITensors.op(::OpName"N2", ::SiteType"Qudit", s::Index)
  mat = diagm(collect(0:dim(s)-1).^2)
  return itensor(mat, s', s)
end

function ITensors.op(::OpName"AAdag", ::SiteType"Qudit", s::Index)
  mat = diagm(1 => fill(1,dim(s)-1), -1 => fill(1,dim(s)-1))
  return itensor(mat, s', s)
end

function energy(psi, gates)
  Hpsi = mapreduce(+, gates) do h
    return apply(h, psi; cutoff=0)
  end

  H2psi = mapreduce(+, gates) do h
    return apply(h, Hpsi; cutoff=0)
  end

  meanH = inner(psi', Hpsi)
  varH = inner(psi', H2psi) - meanH^2
  return (meanH, varH)
end

let
  N = 4
  D = 5
  cutoff = 0 #1E-14
  tau = 0.001
  ttotal = 2.0
  (J,g) = (-1., 3.)

  ## Ising model ##
  s = siteinds("Qudit", N; dim = D)

  intGates = ITensor[]
  for (j1,j2) in zip(1:N, vcat(2:N,1))
    push!(intGates, J*op("Nabs", s[j1], s[j2]))
  end
  
  flipGates = ITensor[]
  for j in 1:N
    push!(flipGates, g*op("AAdag", s[j]))
  end

  evolGates = exp.( - (tau/2) .* vcat(intGates, flipGates))
  append!(evolGates, reverse(evolGates))

  # psi = MPS(s, "$(div(D-1,2))")
  psi = randomMPS(s)
  data = []

  t = 0
  while true
    println("time t = ", round(t, digits = 1))

    ## observables
    meanN = expect(psi, "N") 
    # absN = expect(psi, "Nabs")
    absN = inner(psi', apply(intGates, psi; cutoff=0))
    varN = expect(psi, "N2") .- meanN.^2
    meanH, varH = energy(psi, intGates)

    push!(data, [t, meanN .- div(D-1,2), absN, varN, meanH, varH])

    psi = apply(evolGates, psi; cutoff)
    normalize!(psi)

    t += tau
    t≈ttotal && break
  end

  @show data[end]

  # df = DataFrame(time = [d[1] for d in data], zPol = [d[2] for d in data]) #, xPol = [d[3] for d in data])
  # name_obs = "data/heisenbergqns_tebd_N=$(N)"
  # CSV.write(name_obs*".csv", df)
end
