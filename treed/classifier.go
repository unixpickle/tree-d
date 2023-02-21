package treed

import "golang.org/x/exp/constraints"

type ClassifierLoss[F constraints.Float] interface {
	LossAndGrad(pred F, target bool) (F, F)
}

type HingeLoss[F constraints.Float] struct{}

func (_ HingeLoss[F]) LossAndGrad(pred F, target bool) (F, F) {
	var loss, grad F
	if pred < 1 && target {
		grad = 1.0
		loss = 1 - pred
	} else if pred > -1 && !target {
		grad = -1.0
		loss = (pred + 1)
	}
	return loss, grad
}

type LinearOptimizer[F constraints.Float, C Coord[F, C]] interface {
	Init(weight C, bias F)
	Step(weightGrad C, biasGrad F) (C, F)
}

type SGDOptimizer[F constraints.Float, C Coord[F, C]] struct {
	LR          F
	WeightDecay F
	Momentum    F

	AnnealIters int

	// Current state
	weight C
	bias   F

	// State updated by the step function.
	iter      int
	momentumW C
	momentumB F
}

func (s *SGDOptimizer[F, C]) Init(weight C, bias F) {
	s.weight = weight
	s.bias = bias
	s.iter = 0
}

func (s *SGDOptimizer[F, C]) Step(weightGrad C, biasGrad F) (C, F) {
	weightGrad = weightGrad.Add(s.weight.Scale(-0.5 * s.WeightDecay))
	biasGrad -= 0.5 * s.bias * s.WeightDecay

	if s.iter == 0 {
		s.momentumW, s.momentumB = weightGrad, biasGrad
	} else {
		s.momentumW = s.momentumW.Scale(s.Momentum).Add(weightGrad)
		s.momentumB = s.momentumB*s.Momentum + biasGrad
	}

	// Nesterov momentum (see https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
	weightGrad = weightGrad.Add(s.momentumW.Scale(s.Momentum))
	biasGrad = biasGrad + s.momentumB*s.Momentum

	lr := s.LR
	if s.AnnealIters != 0 {
		lr *= F(s.AnnealIters-s.iter) / F(s.AnnealIters)
	}
	s.weight = s.weight.Add(weightGrad.Scale(lr))
	s.bias += biasGrad * lr
	s.iter++

	return s.weight, s.bias
}

type LinearRegressionResult[F constraints.Float, C Coord[F, C]] struct {
	Weight    C
	Bias      F
	InitLoss  F
	FinalLoss F
	InitAcc   F
	FinalAcc  F
}

func LinearRegression[F constraints.Float, C Coord[F, C]](
	initW C,
	initB F,
	coords []C,
	targets []bool,
	weights []F,
	lossFn ClassifierLoss[F],
	opt LinearOptimizer[F, C],
	iters int,
) *LinearRegressionResult[F, C] {
	w := initW
	b := initB
	opt.Init(w, b)

	var totalWeight F
	for _, x := range weights {
		totalWeight += x
	}
	if totalWeight == 0 {
		return &LinearRegressionResult[F, C]{
			Weight: w,
			Bias:   b,
		}
	}
	meanScale := 1 / totalWeight

	var initLoss, finalLoss, initAcc, finalAcc F

	for iter := 0; iter < iters; iter++ {
		var weightGrad C
		var biasGrad F
		var totalLoss F
		var acc F
		for i, c := range coords {
			target := targets[i]
			weight := weights[i] * meanScale
			if weight == 0 {
				continue
			}
			pred := w.Dot(c) + b
			loss, lossGrad := lossFn.LossAndGrad(pred, target)
			if pred > 0 == target {
				acc += weight
			}
			weightGrad = weightGrad.Add(c.Scale(lossGrad * weight))
			biasGrad += lossGrad * weight
			totalLoss += loss * weight
		}

		if iter == 0 {
			initLoss = totalLoss
			initAcc = acc
		} else if iter == iters-1 {
			finalLoss = totalLoss
			finalAcc = acc
		}

		w, b = opt.Step(weightGrad, biasGrad)
	}

	return &LinearRegressionResult[F, C]{
		Weight:    w,
		Bias:      b,
		InitLoss:  initLoss,
		FinalLoss: finalLoss,
		InitAcc:   initAcc,
		FinalAcc:  finalAcc,
	}
}
