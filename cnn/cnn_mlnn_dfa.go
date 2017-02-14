package cnn

import (
	"github.com/jonysugianto/dfa_nn/mlnn"
	"github.com/jonysugianto/mathlib/matrix"
)

type CnnMlpDfa struct {
	CnnLayers *MultiLayerCnnDfa
	MlpLayers *mlnn.MLNN
}

func (this *CnnMlpDfa) Init() {
	this.CnnLayers.Init()
	this.MlpLayers.Init()
}

func (this *CnnMlpDfa) Outputs() *matrix.Vector {
	return this.MlpLayers.Outputs()
}

func (this *CnnMlpDfa) Running(input *matrix.MatrixFloat) {
	this.CnnLayers.Activate(input)
	this.MlpLayers.Running(this.CnnLayers.Outputs())
}

func (this *CnnMlpDfa) Learning(input *matrix.MatrixFloat, target *matrix.Vector) *matrix.Vector {
	this.Running(input)
	var target_error = this.MlpLayers.ComputeTargetError(target)
	this.MlpLayers.ComputeDFAError(target_error)
	this.MlpLayers.Update(this.CnnLayers.Outputs())

	this.CnnLayers.ComputeDFAError(target_error)
	this.CnnLayers.Compute_Dw_Dt()
	this.CnnLayers.Update_W_T()
	return target_error
}
