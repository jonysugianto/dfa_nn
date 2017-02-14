package cnn

import (
	"github.com/jonysugianto/dfa_nn/mlnn"
)

type CnnMlpDfaBuilder struct {
	MlCnnBuilder *MultiLayerCnnBuilder
	MlnnBuilder  *mlnn.MLNNBuilder
}

func CreateBuilder() *CnnMlpDfaBuilder {
	var ret = new(CnnMlpDfaBuilder)
	ret.MlCnnBuilder = new(MultiLayerCnnBuilder)
	ret.MlnnBuilder = new(mlnn.MLNNBuilder)
	return ret
}

func (this *CnnMlpDfaBuilder) Build() (*CnnMlpDfa, error) {
	var ret = new(CnnMlpDfa)
	ret.CnnLayers, _ = this.MlCnnBuilder.Build()
	ret.MlpLayers, _ = this.MlnnBuilder.Build()
	return ret, nil
}
