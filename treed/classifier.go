package treed

import (
	"math"

	"golang.org/x/exp/constraints"
)

type ClassifierLoss[F constraints.Float] interface {
	// LossAndGrad computes the loss and/or gradient of the loss.
	//
	// Both lossOut or gradOut may be passed, or one may be nil, but at least
	// one of the two should be non-nil.
	LossAndGrad(preds []F, targets []bool, lossOut, gradOut []F)
}

type HingeLoss[F constraints.Float] struct{}

func (_ HingeLoss[F]) LossAndGrad(preds []F, targets []bool, lossOut, gradOut []F) {
	if len(preds) != len(targets) {
		panic("mismatching input sizes")
	}
	if lossOut == nil && gradOut == nil {
		panic("must provide lossOut or gradOut")
	}
	if lossOut != nil && len(lossOut) != len(preds) {
		print("incorrect lossOut size")
	}
	if gradOut != nil && len(gradOut) != len(preds) {
		print("incorrect gradOut size")
	}
	for i, p := range preds {
		t := targets[i]
		if p < 1 && t {
			if gradOut != nil {
				gradOut[i] = -1.0
			}
			if lossOut != nil {
				lossOut[i] = 1 - p
			}
		} else if p > -1 && !t {
			if gradOut != nil {
				gradOut[i] = 1.0
			}
			if lossOut != nil {
				lossOut[i] = (p + 1)
			}
		} else {
			if gradOut != nil {
				gradOut[i] = 0
			}
			if lossOut != nil {
				lossOut[i] = 0
			}
		}
	}
}

func LineSearchScale[F constraints.Float, C Coord[F, C]](
	weight C,
	bias F,
	coords []C,
	targets []bool,
	loss ClassifierLoss[F],
) F {
	outputs := make([]F, len(coords))
	for i, c := range coords {
		outputs[i] = c.Dot(weight) + bias
	}

	// Temporary buffers.
	scaled := make([]F, len(outputs))
	losses := make([]F, len(outputs))

	lossForScale := func(scale F) F {
		for i, x := range outputs {
			scaled[i] = x * scale
		}
		loss.LossAndGrad(scaled, targets, losses, nil)
		var total F
		for _, loss := range losses {
			total += loss
		}
		return total
	}

	// Exponentially increase the scale to find an order-of-magnitude
	// estimate of the optimal scale.
	bestScale := F(1e-5)
	bestLoss := lossForScale(bestScale)
	for scale := bestScale * 2.0; scale < 1e5; scale *= 2 {
		loss := lossForScale(scale)
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			break
		}
		if loss < bestLoss {
			bestLoss = loss
			bestScale = scale
		}
	}

	// Bisection search to find a local minimum.
	minScale := bestScale / 2.0
	maxScale := bestScale * 2.0
	for i := 0; i < 16; i++ {
		s1 := (minScale*0.75 + maxScale*0.25)
		s2 := (minScale*0.25 + maxScale*0.75)
		loss1 := lossForScale(s1)
		loss2 := lossForScale(s2)
		if loss1 < loss2 {
			maxScale = s2
		} else {
			minScale = s1
		}
	}

	return (minScale + maxScale) / 2
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
	s.weight = s.weight.Add(weightGrad.Scale(-lr))
	s.bias += -biasGrad * lr
	s.iter++

	return s.weight, s.bias
}

type LinearClassificationResult[F constraints.Float, C Coord[F, C]] struct {
	Weight    C
	Bias      F
	InitLoss  F
	FinalLoss F
	InitAcc   F
	FinalAcc  F
}

func LinearClassification[F constraints.Float, C Coord[F, C]](
	initW C,
	initB F,
	coords []C,
	targets []bool,
	weights []F,
	lossFn ClassifierLoss[F],
	opt LinearOptimizer[F, C],
	iters int,
) *LinearClassificationResult[F, C] {
	w := initW
	b := initB
	opt.Init(w, b)

	var totalWeight F
	for _, x := range weights {
		totalWeight += x
	}
	if totalWeight == 0 {
		return &LinearClassificationResult[F, C]{
			Weight: w,
			Bias:   b,
		}
	}
	meanScale := 1 / totalWeight

	var initLoss, finalLoss, initAcc, finalAcc F

	// Temporary buffers.
	preds := make([]F, len(coords))
	losses := make([]F, len(coords))
	grads := make([]F, len(coords))

	for iter := 0; iter < iters; iter++ {
		var acc F
		for i, c := range coords {
			pred := w.Dot(c) + b
			preds[i] = pred
			if pred > 0 == targets[i] {
				acc += weights[i]
			}
		}

		lossFn.LossAndGrad(preds, targets, losses, grads)

		var weightGrad C
		var biasGrad F
		var totalLoss F
		for i, c := range coords {
			weight := weights[i] * meanScale
			loss := losses[i]
			grad := grads[i]

			weightGrad = weightGrad.Add(c.Scale(grad * weight))
			biasGrad += grad * weight
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

	return &LinearClassificationResult[F, C]{
		Weight:    w,
		Bias:      b,
		InitLoss:  initLoss,
		FinalLoss: finalLoss,
		InitAcc:   initAcc,
		FinalAcc:  finalAcc,
	}
}
