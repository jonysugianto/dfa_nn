package cnn

type MultiLayerCnnBuilder struct {
	Configs      []ConvConf
	Learningrate float64
}

func (this *MultiLayerCnnBuilder) AddCnnMaxpoolLayer(config ConvConf) *MultiLayerCnnBuilder {
	this.Configs = append(this.Configs, config)
	return this
}

func (this *MultiLayerCnnBuilder) SetLearningrate(lr float64) *MultiLayerCnnBuilder {
	this.Learningrate = lr
	return this
}

func (this *MultiLayerCnnBuilder) Build() (*MultiLayerCnnDfa, error) {
	var ret = new(MultiLayerCnnDfa)
	var size = len(this.Configs)
	for i := 0; i < size; i++ {
		var hl = CreateConvMaxpoolLayer(this.Configs[i])
		ret.Layers = append(ret.Layers, hl)
	}
	ret.Learningrate = this.Learningrate
	return ret, nil
}
