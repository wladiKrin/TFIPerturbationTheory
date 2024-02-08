using Revise
using Statistics
using CSV, DataFrames
using LinearAlgebra
using Observers

using ITensors
using ITensorTDVP

include("hilbertspace.jl")

function Ising(;L, g, J = -1, d)
  ampo = OpSum()
  for (j1,j2) in zip(1:L, vcat(2:L, 1))
    ampo += (g,"X", j1) 

    for s in -d:d
      for k in -d:d
          ampo += (-2*J*abs(s-k), "Proj$(s)", j1, "Proj$(k)", j2)
      end
    end
  end

  return ampo
end

function current_time(; current_time, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return abs(current_time)
  end
  return nothing
end

function measure_N(; psi, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return mean(expect(psi, "N"))
  end
  return nothing
end

function measure_Imb(; psi, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return mean(expect(psi, "Imb"))
  end
  return nothing
end

function measure_N2(; psi, bond, half_sweep)
  if bond == 1 && half_sweep == 2
    return mean(expect(psi, "N2"))
  end
  return nothing
end

Ls = (8,16,20)
bondDims = (16,32)
gs = [-0.25,-0.5,-0.75,-1.0,-1.25,-1.5,-1.75,-2.0]

let
  N = 8 #length of lattice
  D = 4 #max Sz component
  (J,g,h) = (-1.,-0.5,-0.)
  maxDim = 16


  tmax = 10.
  dt = 0.1

  ## Ising model ##
  s = siteinds("SpinN", N; nz = D)
  
  # Model MPO
  model = Ising(L=N, g=g, J = J, d=D)

  # Make MPO
  H = ITensors.MPO(model, s)

  function measure_En(; psi, bond, half_sweep)
    if bond == 1 && half_sweep == 2
      return real(inner(psi', H,psi))
    end
    return nothing
  end
  
  psi = MPS(s, "0")

  obs = observer(
    "time" => current_time, "energy" => measure_En, "N" => measure_N, "imb" => measure_Imb, "N2" => measure_N2,
  )

  phi = tdvp(
    H,
    -im * tmax,
    psi;
    time_step = -im * dt,
    reverse_step=true,
    normalize=false,
    maxdim=maxDim,
    cutoff=1e-10,
    outputlevel=1,
    (observer!)=obs,
  )

  df = DataFrame(
    t   = obs.time, 
    en  = obs.energy, 
    N   = obs.N, 
    imb = obs.imb, 
    N2  = obs.N2, 
  )
  CSV.write("../../data/obs_hardcoreBosons_mps_L=$(N)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).csv", df)
end
