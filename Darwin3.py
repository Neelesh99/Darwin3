# -*- coding: latin-1 -*-
#
#       Darwin3.py
#
#  Copyright 2020 Neelesh Ravichandran <neelesh.ravichandran1999@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#

# Backend Start
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import rmsprop
from keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, AveragePooling2D, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import Adam

import Exceptions as exp

import random
import math
import numpy as np
import Exceptions as exp

# Defining Global Variables #
# HyperParameters #
# Learning Rate #
LR_PLATEAU = 0
LR_SCHEDULER = 0
# Optimizer #
ADAM = 0
SGD = 0
ADADELTA = 0
# Regularizers #
NONE = 0
# Gene Types #
CONVOLUTIONAL_ACTIVATION = 0
POOLING = 1
DENSE_ACTIVATION = 2
BATCH_NORMALISATION = 3
DROPOUT = 4
# ConvAct Gene parameters #
FILTER_COUNT = 0
CONV_DIM = 1
PADDING_TYPE = 2
CONV_STRIDE = 3
CONV_ACT = 4
conv_max_min = {
    FILTER_COUNT: (1, 129),
    CONV_DIM: (1, 3),
    PADDING_TYPE: (0, 1),
    CONV_STRIDE: (1, 3),
    CONV_ACT: (0, 3)
}
# Pool Gene parameters #
POOLING_TYPE = 0
POOL_DIM = 1
POOL_STRIDE = 2
pool_max_min = {
    POOLING_TYPE: (0, 1),
    POOL_DIM: (1, 3),
    POOL_STRIDE: (1, 3)
}
# Dense Gene parameters #
NO_NODES = 0
DENSE_ACT = 1
dense_max_min = {
    NO_NODES: (1, 20),
    DENSE_ACT: (0, 3)
}
# Batch Normalisation #
batch_min_max = {

}
# Dropout Layer #
DROPOUT_RATE = 0
drop_min_max = {
    DROPOUT_RATE: (0, 100)
}
# Mutation Rate #
MUTATIONRATE = 0.1
# Epoch count
EPOCHS = 10
# Network Mode
CLASSIFICATION = 0
REGRESSION = 1


class Gene:
    Type = -1

    def __init__(self):
        Type = -1

    def get_type(self):
        return self.Type

    def check_max_min(self, tupleMaxMin, val):
        if val < tupleMaxMin[0]:
            return tupleMaxMin[0]
        elif val > tupleMaxMin[1]:
            return tupleMaxMin[1]
        else:
            return val


class ConvActGene(Gene):
    FilterCount = 0
    ConvDim = 0
    PaddingType = 0
    ConvStride = 0
    ConvAct = 0

    def __init__(self):
        super().__init__()
        self.Type = CONVOLUTIONAL_ACTIVATION

    def setDataGroup(self, DataList):
        try:
            if len(DataList) != 5:
                raise exp.GeneError(
                    "Expected Gene Data of length 5 for Convolutional Gene, got {}".format(len(DataList)))
            else:
                for i in range(len(DataList)):
                    self.setIndividual(i, DataList[i])
        except exp.GeneError as ge:
            print("GeneError occured in Convolutional Gene construction. \n message: {}".format(ge))

    def setIndividual(self, Parameter, data):
        val = self.check_max_min(conv_max_min.get(Parameter), data)
        if (Parameter == FILTER_COUNT):
            self.FilterCount = val
        elif (Parameter == CONV_DIM):
            self.ConvDim = val
        elif (Parameter == PADDING_TYPE):
            self.PaddingType = val
        elif (Parameter == CONV_STRIDE):
            self.ConvStride = val
        else:
            self.ConvAct = val

    def getData(self):
        return [self.FilterCount, self.ConvDim, self.PaddingType, self.ConvStride, self.ConvAct]

    def get_min_max(self, Parameter):
        return conv_max_min.get(Parameter)


class PoolGene(Gene):
    PoolingType = 0
    PoolingDim = 0
    PoolingStride = 0

    def __init__(self):
        super().__init__()
        self.Type = POOLING

    def setDataGroup(self, DataList):
        try:
            if len(DataList) != 3:
                raise exp.GeneError("Expected Gene Data of length 3 for Pooling gene, got {}".format(len(DataList)))
            else:
                for i in range(len(DataList)):
                    self.setIndividual(i, DataList[i])
        except exp.GeneError as ge:
            print("GeneError occured in Pooling Gene construction. \n message: {}".format(ge))

    def setIndividual(self, Parameter, data):
        val = self.check_max_min(pool_max_min.get(Parameter), data)
        if (Parameter == POOLING_TYPE):
            self.PoolingType = val
        elif (Parameter == POOL_DIM):
            self.PoolingDim = val
        else:
            self.PoolingStride = val

    def getData(self):
        return [self.PoolingType, self.PoolingDim, self.PoolingStride]

    def get_min_max(self, Parameter):
        return pool_max_min.get(Parameter)


class DenseGene(Gene):
    NodeCount = 0
    DenseAct = 0

    def __init__(self):
        super().__init__()
        self.Type = DENSE_ACTIVATION

    def setDataGroup(self, DataList):
        try:
            if len(DataList) != 2:
                raise exp.GeneError("Expected Gene Data of length 2 for Dense gene, got {}".format(len(DataList)))
            else:
                for i in range(len(DataList)):
                    self.setIndividual(i, DataList[i])
        except exp.GeneError as ge:
            print("GeneError occured in Dense Gene construction. \n message: {}".format(ge))

    def setIndividual(self, Parameter, data):
        val = self.check_max_min(dense_max_min.get(Parameter), data)
        if (Parameter == NO_NODES):
            self.NodeCount = val
        elif (Parameter == DENSE_ACT):
            self.DenseAct = val

    def getData(self):
        return [self.NodeCount, self.DenseAct]

    def get_min_max(self, Parameter):
        return dense_max_min.get(Parameter)


class BatchNormGene(Gene):
    def __init__(self):
        super().__init__()
        self.Type = BATCH_NORMALISATION

    def setDataGroup(self, DataList):
        try:
            if len(DataList) != 0:
                raise exp.GeneError("Expected Gene Data of length 3 for Pooling gene, got {}".format(len(DataList)))
            else:
                for i in range(len(DataList)):
                    self.setIndividual(i, DataList[i])
        except exp.GeneError as ge:
            print("GeneError occured in Pooling Gene construction. \n message: {}".format(ge))

    def setIndividual(self, Parameter, data):
        print('Not implemented')

    def getData(self):
        return []

    def get_min_max(self, Parameter):
        return batch_min_max.get(Parameter)


class DropoutGene(Gene):
    DropoutRate = 0

    def __init__(self):
        super().__init__()
        self.Type = DROPOUT

    def setDataGroup(self, DataList):
        try:
            if len(DataList) != 1:
                raise exp.GeneError("Expected Gene Data of length 3 for Pooling gene, got {}".format(len(DataList)))
            else:
                for i in range(len(DataList)):
                    self.setIndividual(i, DataList[i])
        except exp.GeneError as ge:
            print("GeneError occured in Dropout Gene construction. \n message: {}".format(ge))

    def setIndividual(self, Parameter, data):
        val = self.check_max_min(dense_max_min.get(Parameter), data)
        self.DropoutRate = (float(val) / 100.0)

    def getData(self):
        return [self.DropoutRate]

    def get_min_max(self, Parameter):
        return drop_min_max.get(Parameter)


class Chromosome:
    Genes = []

    def __init__(self, gene_list):
        self.Genes = gene_list

    def getLength(self):
        return len(self.Genes)

    def getGenes(self):
        return self.Genes


class OccularChromosome(Chromosome):
    def __init__(self, gene_list):
        super().__init__(gene_list)


class CortexChromosome(Chromosome):
    def __init__(self, gene_list):
        super().__init__(gene_list)


class Genome:
    Occular = None
    Cortex = None

    def __init__(self, Occ, Cor):
        self.Occular = Occ
        self.Cortex = Cor

    def getOccular(self):
        return self.Occular

    def getCortex(self):
        return self.Cortex


class FusionEngine:
    Genome1 = None
    Genome2 = None
    Accuracy1 = 0
    Accuracy2 = 0

    def __init__(self, gen1, acc1, gen2, acc2):
        self.Genome1 = gen1
        self.Genome2 = gen2
        self.Accuracy1 = acc1
        self.Accuracy2 = acc2

    def floater(self, x):
        return float(x)

    def get_next(self, Genes, pointer, Type):
        if pointer == -1:
            return (Genes[0], -1)
        while pointer < len(Genes):
            if Genes[pointer].get_type() == Type:
                pointer_next = pointer + 1
                return (Genes[pointer], pointer_next)
            else:
                pointer += 1
        return (Genes[0], -1)

    def baseMix(self, Base1, Base2):
        Tot = self.Accuracy1 + self.Accuracy2
        L1 = self.floater(Base1) * self.Accuracy1 + self.floater(Base2) * self.Accuracy2
        L2 = round(L1 / Tot)
        return int(L2)

    def mutate(self, Val, min_max):
        if random.random() < MUTATIONRATE:
            return random.randint(min_max[0], min_max[1])
        else:
            return Val

    def mixConv(self, Gene1, Gene2):
        Temp_Filter_Count = self.baseMix(Gene1.FilterCount, Gene2.FilterCount)
        Temp_Filter_Count = self.mutate(Temp_Filter_Count, conv_max_min.get(FILTER_COUNT))
        Temp_Conv_Dim = self.baseMix(Gene1.ConvDim, Gene2.ConvDim)
        Temp_Conv_Dim = self.mutate(Temp_Conv_Dim, conv_max_min.get(CONV_DIM))
        Temp_Padding_Type = self.baseMix(Gene1.PaddingType, Gene2.PaddingType)
        Temp_Padding_Type = self.mutate(Temp_Padding_Type, conv_max_min.get(PADDING_TYPE))
        Temp_Conv_Stride = self.baseMix(Gene1.ConvStride, Gene2.ConvStride)
        Temp_Conv_Stride = self.mutate(Temp_Conv_Stride, conv_max_min.get(CONV_STRIDE))
        Temp_Conv_Act = self.baseMix(Gene1.ConvAct, Gene2.ConvAct)
        Temp_Conv_Act = self.mutate(Temp_Conv_Act, conv_max_min.get(CONV_ACT))

        ChildGene = ConvActGene()
        ChildGene.setDataGroup([Temp_Filter_Count, Temp_Conv_Dim, Temp_Padding_Type, Temp_Conv_Stride, Temp_Conv_Act])
        return ChildGene

    def mixPool(self, Gene1, Gene2):
        Temp_Pooling_Type = self.baseMix(Gene1.PoolingType, Gene2.PoolingType)
        Temp_Pooling_Type = self.mutate(Temp_Pooling_Type, pool_max_min.get(POOLING_TYPE))
        Temp_Pooling_Dim = self.baseMix(Gene1.PoolingDim, Gene2.PoolingDim)
        Temp_Pooling_Dim = self.mutate(Temp_Pooling_Dim, pool_max_min.get(POOL_DIM))
        Temp_Pooling_Stride = self.baseMix(Gene1.PoolingStride, Gene2.PoolingStride)
        Temp_Pooling_Stride = self.mutate(Temp_Pooling_Stride, pool_max_min.get(POOL_STRIDE))

        ChildGene = PoolGene()
        ChildGene.setDataGroup([Temp_Pooling_Type, Temp_Pooling_Dim, Temp_Pooling_Stride])
        return ChildGene

    def mixDense(self, Gene1, Gene2):
        Temp_Node_Count = self.baseMix(Gene1.NodeCount, Gene2.NodeCount)
        Temp_Node_Count = self.mutate(Temp_Node_Count, dense_max_min.get(NO_NODES))
        Temp_Dense_Act = self.baseMix(Gene1.DenseAct, Gene2.DenseAct)
        Temp_Dense_Act = self.mutate(Temp_Dense_Act, dense_max_min.get(DENSE_ACT))

        ChildGene = DenseGene()
        ChildGene.setDataGroup([Temp_Node_Count, Temp_Dense_Act])
        return ChildGene

    def mixBatchNormalisation(self, Gene1, Gene2):
        ChildGene = BatchNormGene()
        return ChildGene

    def mixDropout(self, Gene1, Gene2):
        Temp_Dropout = self.baseMix(Gene1.DropoutRate * 100, Gene2.DropoutRate * 100)
        Temp_Dropout = self.mutate(Temp_Dropout, drop_min_max.get(DROPOUT_RATE))

        ChildGene = DropoutGene()
        ChildGene.setDataGroup([Temp_Dropout])
        return ChildGene

    def mixOccular(self):
        Occular1 = self.Genome1.getOccular().getGenes()
        Occular2 = self.Genome2.getOccular().getGenes()
        top_pointer = 0
        bottom_cv_pointer = 0
        bottom_pool_pointer = 0
        bottom_batch_pointer = 0
        bottom_drop_pointer = 0
        ChildChromosomePrototype = []
        End = False
        while not End:
            Gene1 = Occular1[top_pointer]
            t1 = Gene1.get_type()
            if t1 == CONVOLUTIONAL_ACTIVATION:
                (Gene2, Index2) = self.get_next(Occular2, bottom_cv_pointer, CONVOLUTIONAL_ACTIVATION)
                if (Index2 == -1):
                    ChildChromosomePrototype.append(Gene1)
                else:
                    ChildChromosomePrototype.append(self.mixConv(Gene1, Gene2))
                top_pointer += 1
                bottom_cv_pointer = Index2
            elif t1 == POOLING:
                (Gene2, Index2) = self.get_next(Occular2, bottom_pool_pointer, POOLING)
                if (Index2 == -1):
                    ChildChromosomePrototype.append(Gene1)
                else:
                    ChildChromosomePrototype.append(self.mixPool(Gene1, Gene2))
                top_pointer += 1
                bottom_pool_pointer = Index2
            elif t1 == BATCH_NORMALISATION:
                (Gene2, Index2) = self.get_next(Occular2, bottom_batch_pointer, BATCH_NORMALISATION)
                if (Index2 == -1):
                    ChildChromosomePrototype.append(Gene1)
                else:
                    ChildChromosomePrototype.append(self.mixBatchNormalisation(Gene1, Gene2))
                top_pointer += 1
                bottom_batch_pointer = Index2
            elif t1 == DROPOUT:
                (Gene2, Index2) = self.get_next(Occular2, bottom_drop_pointer, DROPOUT)
                if (Index2 == -1):
                    ChildChromosomePrototype.append(Gene1)
                else:
                    ChildChromosomePrototype.append(self.mixDropout(Gene1, Gene2))
                top_pointer += 1
                bottom_drop_pointer = Index2
            if (top_pointer >= len(Occular1)):
                End = True
        return ChildChromosomePrototype

    def mixCortex(self):
        Cortex1 = self.Genome1.getCortex().getGenes()
        Cortex2 = self.Genome2.getCortex().getGenes()
        top_pointer = 0
        bottom_dense_pointer = 0
        bottom_drop_pointer = 0
        ChildChromosomePrototype = []
        End = False
        while not End:
            Gene1 = Cortex1[top_pointer]
            t1 = Gene1.get_type()
            if t1 == DENSE_ACTIVATION:
                (Gene2, Index2) = self.get_next(Cortex2, bottom_dense_pointer, DENSE_ACTIVATION)
                if (Index2 == -1):
                    ChildChromosomePrototype.append(Gene1)
                else:
                    ChildChromosomePrototype.append(self.mixDense(Gene1, Gene2))
                top_pointer += 1
                bottom_dense_pointer = Index2
            elif t1 == DROPOUT:
                (Gene2, Index2) = self.get_next(Cortex2, bottom_drop_pointer, DROPOUT)
                if (Index2 == -1):
                    ChildChromosomePrototype.append(Gene1)
                else:
                    ChildChromosomePrototype.append(self.mixDropout(Gene1, Gene2))
                top_pointer += 1
                bottom_drop_pointer = Index2
            if top_pointer >= len(Cortex1):
                End = True
        return ChildChromosomePrototype

    def TotalMix(self):
        OccularPrototype = self.mixOccular()
        CortexPrototype = self.mixCortex()
        OccularChrom = OccularChromosome(OccularPrototype)
        CortexChrom = CortexChromosome(CortexPrototype)
        ChildGenome = Genome(OccularChrom, CortexChrom)
        return ChildGenome


def classifcation_sahara(model, x_t, y_t, x_te, y_te):
    opt = rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_t, y_t, batch_size=32, epochs=EPOCHS, verbose=0)
    score = model.evaluate(x_te, y_te, verbose=0)
    return score


def regression_serengeti(model, x_t, y_t, x_v, y_v):
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
    model.fit(x_t, y_t, validation_data=(x_v, y_v), epochs=100, batch_size=32)
    prices_predicted = model.predict(x_v).flatten()
    diff = ((prices_predicted - y_v) / y_v)
    error_mean = np.mean(np.abs(diff))
    return [0, (1 - error_mean)]


class Environment:
    XTrain = []
    XTest = []
    YTrain = []
    YTest = []
    Mode = None

    def __init__(self, x_train1, y_train1, x_test1, y_test1, mode):
        self.Mode = mode
        self.XTrain = x_train1
        self.YTrain = y_train1
        self.XTest = x_test1
        self.YTest = y_test1

    def run(self, models):
        score_vector = []
        for model_x in models:
            score = None
            # Insert classifier
            if self.Mode == CLASSIFICATION:
                score = classifcation_sahara(model_x, self.XTrain, self.YTrain, self.XTest, self.YTest)
            elif self.Mode == REGRESSION:
                score = regression_serengeti(model_x, self.XTrain, self.YTrain, self.XTest, self.YTest)
            score_vector.append(score[1])
        return score_vector


class Translator:
    First_Conv = False
    Mode = None

    def __init__(self, mode):
        self.First_Conv = True
        self.Mode = mode

    def getLayer(self, gene, model, input_shape):
        Type = gene.get_type()
        if Type == CONVOLUTIONAL_ACTIVATION:
            return self.getConvLayer(gene, model, input_shape)
        elif Type == POOLING:
            return self.getPoolLayer(gene, model)
        elif Type == DENSE_ACTIVATION:
            return self.getDenseLayer(gene, model)
        elif Type == BATCH_NORMALISATION:
            return self.getBatchNormalisationLayer(gene, model)
        elif Type == DROPOUT:
            return self.getDropoutLayer(gene, model)

    def getPadding(self, index):
        if index == 1:
            return 'same'
        else:
            print('valid')
            return 'valid'

    def getActivation(self, index):
        if index == 0:
            return 'sigmoid'
        elif index == 1:
            return 'tanh'
        elif index == 2:
            return 'relu'
        elif index == 3:
            return 'hard_sigmoid'

    def getPoolType(self, Index):
        if Index == 0:
            return True
        else:
            return False

    def getConvLayer(self, gene, model, input_shape):
        try:
            if self.First_Conv:
                model.add(
                    Conv2D(gene.FilterCount, (gene.ConvDim, gene.ConvDim), padding=self.getPadding(gene.PaddingType),
                           strides=gene.ConvStride, input_shape=input_shape))
                model.add(Activation(self.getActivation(gene.ConvAct)))
                self.First_Conv = False
            else:
                model.add(
                    Conv2D(gene.FilterCount, (gene.ConvDim, gene.ConvDim), padding=self.getPadding(gene.PaddingType),
                           strides=gene.ConvStride))
                model.add(Activation(self.getActivation(gene.ConvAct)))
        except ValueError as err:
            model.add(Conv2D(gene.FilterCount, (gene.ConvDim, gene.ConvDim), padding=self.getPadding(gene.PaddingType),
                             strides=1, input_shape=input_shape))
        return model

    def getPoolLayer(self, gene, model):
        try:
            if self.getPoolType(gene.PoolingType):
                model.add(MaxPooling2D(pool_size=(gene.PoolingDim, gene.PoolingDim), strides=gene.PoolingStride))
            else:
                model.add(AveragePooling2D(pool_size=(gene.PoolingDim, gene.PoolingDim), strides=gene.PoolingStride))
        except ValueError as err:
            pass

        return model

    def getDenseLayer(self, gene, model):
        model.add(Dense(gene.NodeCount))
        model.add(Activation(self.getActivation(gene.DenseAct)))
        return model

    def getBatchNormalisationLayer(self, gene, model):
        if (self.Mode == CLASSIFICATION):
            model.add(BatchNormalization())
        else:
            model.add(BatchNormalization(axis=-1))
        return model

    def getDropoutLayer(self, gene, model):
        model.add(Dropout(gene.DropoutRate))
        return model

    def translate(self, genome, input_shape):
        Occular = genome.getOccular().getGenes()
        Cortex = genome.getCortex().getGenes()
        # Input Stage Checks
        try:
            if len(Occular) <= 0:
                raise exp.GenomeError("Occular genome has no genes in it, critical failure")
            else:
                model = Sequential()
                # Input Conv Layer
                for i in Occular:
                    model = self.getLayer(i, model, input_shape)
                model.add(Flatten())
                for i in Cortex:
                    model = self.getLayer(i, model, input_shape)

                if self.Mode == CLASSIFICATION:
                    model.add(Dense(10))
                    model.add(Activation('softmax'))
                elif self.Mode == REGRESSION:
                    model.add(Dense(1))
                    model.add(Activation('relu'))
                return model



        except exp.GenomeError as err:
            print("Translator error, message: {}".format(err))

    def untranslate(self, genome):
        Output_list = []
        Occular = []
        Cortex = []
        OccChrom = genome.getOccular().getGenes()
        CorChrom = genome.getCortex().getGenes()
        for i in OccChrom:
            if i.get_type() == CONVOLUTIONAL_ACTIVATION:
                lst = ['Convolution and Activation']
                lst.extend(i.getData())
                Occular.append(lst)
            elif i.get_type() == POOLING:
                lst = ['Pooling Layer']
                lst.extend(i.getData())
                Occular.append(lst)
            elif i.get_type() == BATCH_NORMALISATION:
                lst = ['Batch Normalisation Layer']
                Occular.append(lst)
            elif i.get_type() == DROPOUT:
                lst = ['Dropout Layer']
                lst.extend(i.getData())
                Occular.append(lst)
        for i in CorChrom:
            if i.get_type() == DENSE_ACTIVATION:
                lst = ['Dense Layer with Activation']
                lst.extend(i.getData())
                Cortex.append(lst)
            elif i.get_type() == DROPOUT:
                lst = ['Dropout Layer']
                lst.extend(i.getData())
                Occular.append(lst)
        Output_list = [Occular, Cortex]
        return Output_list

# =============== End of Backend =================== #
# =============== Front end Begin ==================#

#Engine
#Genetic Engine
class Genetic_Engine:
    Env = None
    Population = []
    Genome_List = []
    Initial_Shape = None
    Mode = None
    def __init__(self, initialPopulation, X_Train, Y_Train, X_Test, Y_Test, mode):
        self.Mode = mode
        self.Env = Environment(X_Train, Y_Train, X_Test, Y_Test, self.Mode)
        self.Initial_Shape = X_Train.shape[1:]
        for (OccGenes, CorGenes) in initialPopulation:
            OccPrototype = []
            for i in OccGenes:
                OccPrototype.append(self.Gene_Builder(i))
            CorPrototype = []
            for i in CorGenes:
                CorPrototype.append(self.Gene_Builder(i))
            Occular = OccularChromosome(OccPrototype)
            Cortex = CortexChromosome(CorPrototype)
            genome = Genome(Occular, Cortex)
            self.Genome_List.append(genome)
        for i in self.Genome_List:
            Z = Translator(self.Mode)
            self.Population.append(Z.translate(i,self.Initial_Shape))
    def Gene_Builder(self, InfList):
        GeneType = InfList[0]
        try:
            if GeneType > 4 or GeneType < 0:
                raise exp.GeneError("Incorrect Gene type input in Gene Builder")
            else:
                if GeneType == CONVOLUTIONAL_ACTIVATION:
                    geneTemp = ConvActGene()
                    geneTemp.setDataGroup(InfList[1:])
                elif GeneType == POOLING:
                    geneTemp = PoolGene()
                    geneTemp.setDataGroup(InfList[1:])
                elif GeneType == DENSE_ACTIVATION:
                    geneTemp = DenseGene()
                    geneTemp.setDataGroup(InfList[1:])
                elif GeneType == BATCH_NORMALISATION:
                    geneTemp = BatchNormGene()
                elif GeneType == DROPOUT:
                    geneTemp = DropoutGene()
                    geneTemp.setDataGroup(InfList[1:])
                return geneTemp
        except exp.GeneError as err:
            print(err)

    def run(self, no_of_generations):
        scores = []
        print("Welcome to Darwin2, running genetic optimiser, generation count = {}".format(no_of_generations))
        for i in range(no_of_generations):
            print("Generation {} seeded, waiting for run".format(i))
            scores = self.Env.run(self.Population) #scores is
            FitnessScores = list(zip(self.Genome_List, scores))
            FitnessScores.sort(key=lambda tup: -tup[1])
            popSize = len(self.Population)
            NextGeneration = []
            NextGeneration.append(FitnessScores[0][0])
            print("Generation {} succesfully evaluated, top score = {}, attempting Mixer protocol".format(i, FitnessScores[0][1]))
            for k in range(1, popSize//2):
                bestGenome = FitnessScores[0][0]
                mateGenome = FitnessScores[k][0]
                Fuse = FusionEngine(bestGenome, FitnessScores[0][1], mateGenome, FitnessScores[k][1])
                ChildGenome = Fuse.TotalMix()
                NextGeneration.append(ChildGenome)
            if len(NextGeneration) < len(self.Genome_List):
                for l in range(2, popSize//4):
                    bestGenome = FitnessScores[1][0]
                    mateGenome = FitnessScores[l][0]
                    Fuse = FusionEngine(bestGenome, FitnessScores[1][1], mateGenome, FitnessScores[l][1])
                    ChildGenome = Fuse.TotalMix()
                    NextGeneration.append(ChildGenome)
                j = len(FitnessScores) - 1
                while len(NextGeneration) < len(self.Genome_List):
                    NextGeneration.append(FitnessScores[j][0]) #Adds to Diversity
                    j -= 1
            self.Genome_List = NextGeneration
            self.Population = []
            for p in self.Genome_List:
                print("Init Mode: {}".format(self.Mode))
                Z = Translator(self.Mode)
                print(self.Initial_Shape)
                self.Population.append(Z.translate(p, self.Initial_Shape))
            print("Generation {} completed, with accuracy {}".format(i, FitnessScores[0][1]))
        scores = self.Env.run(self.Population)
        FitnessScores = list(zip(self.Population, scores, self.Genome_List))
        FitnessScores.sort(key=lambda tup: -tup[1])
        Z = Translator(self.Mode)
        Arch = Z.untranslate(FitnessScores[0][2])
        print("Darwin2 engine run, final evaluated test score = {}".format(FitnessScores[0][1]))
        print("Winning Architecture: ")
        print(Arch)
        return FitnessScores[0][0]
    def getPlainGeneration(self):
        FinalList = []
        for i in self.Genome_List:
            Z = Translator(self.Mode)
            FinalList.append(Z.untranslate(i))
        return FinalList

# =========== End of Darwin Engine ===================== #