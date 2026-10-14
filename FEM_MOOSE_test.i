# inspired by https://mooseframework.inl.gov/source/materials/ComputeThermalExpansionEigenstrain.html

# This test involves only thermal expansion strains

[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 100
  ny = 100
[]

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[AuxVariables]
  [./temp]
    initial_condition = 0.0
  [../]
[]

[Functions]
  [./temperature_load]
    type = ParsedFunction
    expression = 0 + t
  [../]
[]

[Physics]
  [SolidMechanics]
    [QuasiStatic]
      [./all]
        strain = SMALL
        incremental = true
        add_variables = true
        eigenstrain_names = eigenstrain
      [../]
    [../]
  [../]
[]

[AuxKernels]
  [./tempfuncaux]
    type = FunctionAux
    variable = temp
    function = temperature_load
  [../]
[]

[BCs]
  [./x_left]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0.0
  [../]
  [./x_right]
    type = DirichletBC
    variable = disp_x
    boundary = right
    value = 0.0
  [../]
  [./y_bot]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0.0
  [../]
  [./y_top]
    type = DirichletBC
    variable = disp_y
    boundary = top
    value = 0.0
  [../]
[]

[Materials]
  [./elasticity_tensor]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 2.1e5
    poissons_ratio = 0.3
  [../]
  [./small_stress]
    type = ComputeFiniteStrainElasticStress
  [../]
  [./thermal_expansion_strain]
    type = ComputeThermalExpansionEigenstrain
    stress_free_temperature = 0
    thermal_expansion_coeff = 1.5
    temperature = temp
    eigenstrain_name = eigenstrain
  [../]
[]

[Executioner]
  type = Transient
  solve_type = 'PJFNK'

  l_max_its = 50
  nl_max_its = 50
  nl_rel_tol = 1e-12
  nl_abs_tol = 1e-10
  l_tol = 1e-9

  start_time = 0
  end_time = 1
  dt = 0.1
[]

[Outputs]
  exodus = true
[]
