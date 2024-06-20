using Pkg; Pkg.activate(".")
using Revise
using Statistics
using CSV, DataFrames, HDF5
using LinearAlgebra
using Observers

using ITensors
# using ITensorNetworks
using ITensorTDVP
using NamedGraphs

include("hilbertspace.jl")

# util function to save data in obs
function savedata(name::String, obs)
    name == "" && return
    @show obs
    h5open(name*".h5", "w") do file    
        # iterate through the fields of obs and append the data to the dataframe
        names = ["sweep","N","Nabs","N2","proj0","proj1","proj2","proj3","proj4","proj5","proj6","proj7","proj8"]
        for n in names
            create_group(file, n)
        
            for (i,data) in enumerate(obs[:,n])
                file[n][string(i)] = data
            end
        end 
        file["energy"]  = obs[:,"energy"]
        file["ent"] = obs[:,"ent"]
    end
end

function Ising(;L, g, J = -1, d)
  ampo = OpSum()
  # for (j1,j2) in zip(1:L, vcat(2:L, 1))
  for (j1,j2) in zip(1:L-1, 2:L)
    # ampo += (g,"X", j1) 
    ampo += (g,"S+", j1) 
    ampo += (g,"S-", j1) 

    for s in -d:d
      for k in -d:d
          ampo += (-2*J*abs(s-k), "Proj$(s)", j1, "Proj$(k)", j2)
      end
    end
  end
  ampo += (g,"S+", L) 
  ampo += (g,"S-", L) 

  return ampo
end

function current_sweep(; sweep)
  return sweep
end

function measure_N(; state)
  res = real.(collect(expect(state, "N")))
  # println("N: $(res)")
  return res
end
function measure_entropy(; state)
  pos = floor(Int,length(state)/2)
  state_ = orthogonalize(state, pos)
  _,S,_ = svd(state_[pos], (commonind(state_[pos-1], state_[pos]), filter(i->hastags(i, "Site"), inds(state_[pos]))))
  return mapreduce(+, 1:dim(S,1)) do i
    p = S[i,i]^2
    return -p*log(p)
  end
end
function measure_Nabs(; state)
  res = real.(collect(expect(state, "Nabs")))
  # println("Nabs: $(mean(res))")
  return res
end
function measure_N2(; state)
  res = real.(collect(expect(state, "N2")))
  # println("N2: $(mean(res))")
  return res
end

function measure_proj0(; state)
    return real.(collect(expect(state, "Proj0")))
end
function measure_proj1(; state)
    return real.(collect(expect(state, "Proj1"))) .+ real.(collect(expect(state, "Proj-1")))
end
function measure_proj2(; state)
    return real.(collect(expect(state, "Proj2"))) .+ real.(collect(expect(state, "Proj-2")))
end
function measure_proj3(; state)
    return real.(collect(expect(state, "Proj3"))) .+ real.(collect(expect(state, "Proj-3")))
end
function measure_proj4(; state)
    return real.(collect(expect(state, "Proj4"))) .+ real.(collect(expect(state, "Proj-4")))
end
function measure_proj5(; state)
    return real.(collect(expect(state, "Proj5"))) .+ real.(collect(expect(state, "Proj-5")))
end
function measure_proj6(; state)
    return real.(collect(expect(state, "Proj6"))) .+ real.(collect(expect(state, "Proj-6")))
end
function measure_proj7(; state)
    return real.(collect(expect(state, "Proj7"))) .+ real.(collect(expect(state, "Proj-7")))
end
function measure_proj8(; state)
    return real.(collect(expect(state, "Proj8"))) .+ real.(collect(expect(state, "Proj-8")))
end

# Ls = (8,16,20,24)
# bondDims = (16,32)
# gs = [-0.25,-0.5,-0.6,-0.75,-0.8,-0.85,-0.9,-1.0,-1.15,-1.25,-1.5]
gs = [-0.25,-0.5,-0.75,-1.,-1.25,-1.5,-1.75,-2.0]

let
  (J,h) = (-1.,-0.)
  N = parse(Int, ARGS[1]) #length of lattice
  maxDim = parse(Int, ARGS[2])
  g = gs[parse(Int, ARGS[3])]
  D = parse(Int, ARGS[4]) #max Sz component

  tmax = 100
  dt = 0.1

  s = siteinds("SpinN", N; nz = D) #, conserve_qns=true)
  
  # Model MPO
  model = Ising(L=N, g=g, J = J, d=D)
  H = MPO(model, s)

  # Make MPS
  # psi = MPS(s, "0")
  psi= randomMPS(s, "0"; linkdims=maxDim)
  for (i,v) in enumerate(psi)
      if i == 1
          mat = fill(0, size(array(psi[i])))
          mat[1,D+1] = 1
          psi[i] = ITensor(mat, inds(psi[i]))
      elseif i == N
          mat = fill(0, size(array(psi[i])))
          mat[D+1,1] = 1
          psi[i] = ITensor(mat, inds(psi[i]))
      else i == N
          mat = fill(0, size(array(psi[i])))
          mat[D+1,1,1] = 1
          psi[i] = ITensor(mat, inds(psi[i]))
      end
  end

  function measure_En(; state)
    res = real(inner(state', H, state))
    println("E: $(res)")
    return res
  end

  obs = observer(
    "sweep" => current_sweep, 
    "energy" => measure_En, 
    "N" => measure_N, 
    "ent" => measure_entropy, 
    "Nabs" => measure_Nabs, 
    "N2" => measure_N2, 
    "proj0" => measure_proj0,
    "proj1" => measure_proj1,
    "proj2" => measure_proj2,
    "proj3" => measure_proj3,
    "proj4" => measure_proj4,
    "proj5" => measure_proj5,
    "proj6" => measure_proj6,
    "proj7" => measure_proj7,
    "proj8" => measure_proj8,
  )

  t = 0 
  # while maxlinkdim(psi) < maxDim
  #     psi = tdvp(
  #       H,
  #       # -im * tmax,
  #       -im * dt,
  #       psi;
  #       nsweeps = 5,
  #       nsite = 2,
  #       reverse_step=true,
  #       normalize=true,
  #       maxdim=maxDim,
  #       cutoff=1e-14,
  #       outputlevel=1,
  #       (step_observer!)=obs,
  #     )
  #     t += 5*dt
  # end

  psi = tdvp(
    H,
    -im * tmax,
    psi;
    nsweeps = floor(Int, (tmax-t)/dt),
    nsite = 1,
    reverse_step=true,
    normalize=false,
    maxdim=maxDim,
    cutoff=1e-10,
    outputlevel=1,
    (step_observer!)=obs,
  )

  # CSV.write("../../data/obs_hardcoreBosons_mps_new_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).csv", df)
  savedata("../../data/obs_hardcoreBosons_mps_nonPeriodic_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax)", obs)
  # statemps = MPS(collect(vertex_data(state)))
  h5open("./ttns/psi_hardcoreBosons_mps_nonPeriodic_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).h5", "w") do file
    write(file, "mps", psi)
  end
end
