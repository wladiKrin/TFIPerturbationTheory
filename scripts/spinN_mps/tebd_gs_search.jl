using Pkg; Pkg.activate(".")
using ITensors
using HDF5, CSV, DataFrames
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

gs = -1 .* collect(0.2:0.2:2.0)

for g in gs
  N = parse(Int, ARGS[1])
  D = parse(Int, ARGS[2])
  cutoff = 1E-10
  cutoff_tebd = 1e-3
  tau = 1e-3
  maxIter = 1e5
  J = -1.

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

  i = 0
  while true
    ## observables
    meanN = expect(psi, "N") 
    absN = inner(psi', apply(intGates, psi; cutoff=0))
    varN = expect(psi, "N2") .- meanN.^2
    meanH, varH = energy(psi, intGates)

    println("time t = ", round(t, digits = 4), ", energy = ", round(meanH, digits = 4), ", energyVar = ", round(varH, digits = 4))

    push!(data, [t, meanN, absN, varN, meanH, varH])

    psi = apply(evolGates, psi; cutoff)
    normalize!(psi)

    (varH < 1) && (tau = 1e-4)
    varH < cutoff_tebd && break
    i += 1
    i>=maxIter && break
  end

  df = DataFrame(time = [d[1] for d in data], meanN = [d[2] for d in data], absN = [d[3] for d in data], varN = [d[4] for d in data], meanH = [d[5] for d in data], varH = [d[6] for d in data], psi = psi)
  name_obs = "../../data/tebd_L=$(N)_locDim=$(D)_J=$(J)_g=$(g)_cutoff=$(cutoff).csv"
  h5open(name_obs*".h5", "w") do file    
    file["time"]  = df[:,"time"]
    file["meanN"] = df[:,"meanH"]
    file["absN"]  = df[:,"absN"]
    file["varN"]  = df[:,"varN"]
    file["meanH"] = df[:,"meanH"]
    file["varH"]  = df[:,"varH"]
  end
end