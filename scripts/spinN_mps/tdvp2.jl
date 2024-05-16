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
  for (j1,j2) in zip(1:L, vcat(2:L, 1))
    # ampo += (g,"X", j1) 
    ampo += (g,"S+", j1) 
    ampo += (g,"S-", j1) 

    for s in -d:d
      for k in -d:d
          ampo += (-2*J*abs(s-k), "Proj$(s)", j1, "Proj$(k)", j2)
      end
    end
  end

  return ampo
end

function current_sweep(; sweep)
  return sweep
end

function measure_N(; psi)
  res = real.(collect(expect(psi, "N")))
  # println("N: $(res)")
  return res
end
function measure_entropy(; psi)
  pos = floor(Int,length(psi)/2)
  psi_ = orthogonalize(psi, pos)
  _,S,_ = svd(psi_[pos], (commonind(psi_[pos-1], psi_[pos]), filter(i->hastags(i, "Site"), inds(psi_[pos]))))
  return mapreduce(+, 1:dim(S,1)) do i
    p = S[i,i]^2
    return -p*log(p)
  end
end
function measure_Nabs(; psi)
  res = real.(collect(expect(psi, "Nabs")))
  # println("Nabs: $(mean(res))")
  return res
end
function measure_N2(; psi)
  res = real.(collect(expect(psi, "N2")))
  # println("N2: $(mean(res))")
  return res
end

function measure_proj0(; psi)
    return real.(collect(expect(psi, "Proj0")))
end
function measure_proj1(; psi)
    return real.(collect(expect(psi, "Proj1"))) .+ real.(collect(expect(psi, "Proj-1")))
end
function measure_proj2(; psi)
    return real.(collect(expect(psi, "Proj2"))) .+ real.(collect(expect(psi, "Proj-2")))
end
function measure_proj3(; psi)
    return real.(collect(expect(psi, "Proj3"))) .+ real.(collect(expect(psi, "Proj-3")))
end
function measure_proj4(; psi)
    return real.(collect(expect(psi, "Proj4"))) .+ real.(collect(expect(psi, "Proj-4")))
end
function measure_proj5(; psi)
    return real.(collect(expect(psi, "Proj5"))) .+ real.(collect(expect(psi, "Proj-5")))
end
function measure_proj6(; psi)
    return real.(collect(expect(psi, "Proj6"))) .+ real.(collect(expect(psi, "Proj-6")))
end
function measure_proj7(; psi)
    return real.(collect(expect(psi, "Proj7"))) .+ real.(collect(expect(psi, "Proj-7")))
end
function measure_proj8(; psi)
    return real.(collect(expect(psi, "Proj8"))) .+ real.(collect(expect(psi, "Proj-8")))
end

# Ls = (8,16,20,24)
# bondDims = (16,32)
# gs = [-0.25,-0.5,-0.6,-0.75,-0.8,-0.85,-0.9,-1.0,-1.15,-1.25,-1.5]
gs = [-0.25,-0.5,-0.75,-1.,-1.25,-1.5,-1.75,-2.0]

let
  N = parse(Int, ARGS[1]) #length of lattice
  D = 8 #max Sz component
  maxDim = parse(Int, ARGS[2])
  (J,h) = (-1.,-0.)
  g = gs[parse(Int, ARGS[3])]

  tmax = 100
  dt = 0.1

  ## Ising model ##
  graph = N #named_path_graph(N)

  s = siteinds("SpinN", graph; nz = D) #, conserve_qns=true)
  
  # Model MPO
  model = Ising(L=N, g=g, J = J, d=D)
  H = MPO(model, s)

  # Make MPS
  # psis = ["0","0","1","2","2","1","0","0"]
  # psis = ["0","0","0","0","0","0","0","0"]
  psi = MPS(s, "0")

  function measure_En(; psi)
    res = real(inner(psi', H, psi))
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
  while maxlinkdim(psi) < maxDim
      psi = tdvp(
        H,
        # -im * tmax,
        -im * dt,
        psi;
        nsweeps = 1,
        nsite = 2,
        reverse_step=true,
        normalize=true,
        maxdim=maxDim,
        cutoff=1e-14,
        outputlevel=1,
        (step_observer!)=obs,
      )
      t += dt
  end

  psi = tdvp(
    H,
    -im * dt,
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

  df = DataFrame(
    t     = obs.sweep .* dt, 
    energy    = obs.energy, 
    entropy  = obs.ent, 
    N     = obs.N, 
    Nabs  = obs.Nabs, 
    N2    = obs.N2, 
    proj0 = obs.proj0, 
    proj1 = obs.proj1, 
    proj2 = obs.proj2, 
    proj3 = obs.proj3, 
    proj4 = obs.proj4, 
    proj5 = obs.proj5, 
    proj6 = obs.proj6, 
    proj7 = obs.proj7, 
    proj8 = obs.proj8, 
  )

  # CSV.write("../../data/obs_hardcoreBosons_mps_new_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).csv", df)
  savedata("../../data/obs_hardcoreBosons_mps_new_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax)", obs)
  # psimps = MPS(collect(vertex_data(psi)))
  h5open("./ttns/psi_hardcoreBosons_mps_new_L=$(N)_Sz=$(D)_g=$(g)_bondDim=$(maxDim)_tmax=$(tmax).h5", "w") do file
    write(file, "mps", psi)
  end
end
