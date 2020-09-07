#pragma once
#include "Network.cpp"
#include "Activations.cpp"
#include "Losses.cpp"
#include "Optimizers.cpp"
#include "Datasets.cpp"

namespace Datasets{};

namespace Optimizers{};

namespace Losses{};

namespace Activations{};

template <
            typename InnerActivationType,
            typename OutputActivationType,
            typename LossType,
            typename OptimizerType
        >
class Network;
