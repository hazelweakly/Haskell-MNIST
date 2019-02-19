{-# Language BangPatterns #-}
{-# Language ScopedTypeVariables #-}

module Neuralnetwork where

import           Data.Bifunctor                 ( bimap )
import           Data.Functor.Identity
import           Data.IDX
import           Data.Maybe                     ( fromMaybe )
-- import           Debug.Trace
import           Control.Monad.Random
import qualified Data.Vector.Unboxed           as V
import           Numeric.LinearAlgebra          ( Matrix
                                                , Vector
                                                , (<>)
                                                , (#>)
                                                , (><)
                                                , (|>)
                                                )
import qualified Numeric.LinearAlgebra         as LA

-- The main crux of my flaw seems to be that my weights increase without bound.
--
-- I've run into this issue _every_ single time I have written up this neural
-- network.  I've rewritten the weight updates countless times now and it's
-- starting to get to the point where my being past the deadline is a bit
-- ridiculous and if I keep it up I won't even get to the second programming
-- assignment at all.
--
-- See the delta functions and the weight update function for the most likely
-- causes of me screwing up.

-- Weights :
--   - biases = n vector of length = num of inputs to that layer
--   - nodes  = n x m where m = num of outputs to that layer
--   => m -> n layer
data Weights = W
  { wBiases :: !(Vector Double) -- length = num of inputs
  , wNodes  :: !(Matrix Double)
  } deriving Show

-- Type aliases offer no type safety but help make type signatures more
-- readable.
type Network    = [Weights]
type Input      = Vector Double
type Target     = Vector Double
type Output     = Vector Double
type Activation = Vector Double
type Prev a     = a -- Cute trick to help make type signatures more readable
type Diff a     = a

-- Here we see that we have, behold, dreaded globals.
-- However, this is fine because
--  a) I don't care
--  b) The reader pattern is overly complicated for this particular usecase
η :: Double -- Learning Rate
η = 0.1

α :: Double -- Momentum Rate
α = 0.9

-- Layer + Network generation
-- Generates a layer of random numbers between -0.05 and 0.05
randLayer :: MonadRandom m => Int -> Int -> m Weights
randLayer i o = do
  r1 :: Int <- getRandom
  r2 :: Int <- getRandom
  let bias    = (LA.randomVector r1 LA.Uniform o - 0.5) / 10
      weights = LA.uniformSample r2 o (replicate i (-0.05, 0.05))
  pure $ W bias weights

-- Generates a layer consisting of all zeroed weights
zeroLayer :: Int -> Int -> Identity Weights
zeroLayer i o = pure $ W (o |> repeat 0) (o >< i $ repeat 0)

-- A network is a list of layers; this simply steps through the list
randNet :: MonadRandom m => Int -> [Int] -> Int -> m Network
randNet i []       o = (:) <$> randLayer i o <*> pure []
randNet i (h : hs) o = (:) <$> randLayer i h <*> randNet h hs o

zeroNet :: Int -> [Int] -> Int -> Identity Network
zeroNet i []       o = (:) <$> zeroLayer i o <*> pure []
zeroNet i (h : hs) o = (:) <$> zeroLayer i h <*> zeroNet h hs o

-- Forward propagate a neural network
-- a #> b is matrix vector product where vector b is Nx1 dimensional
-- i : a 784 length vector
-- wN : (20,784) shaped matrix
runLayer :: Weights -> Input -> Activation
runLayer (W !wB !wN) !i = (wN #> i) + wB
-- runLayer (W !wB !wN) !i = trace "runLayer" $ (wN #> i) + wB

-- Gets the answer from a neural network by propagating the input forward
runNet :: Network -> Input -> Output
runNet !n !i = last $ runNet' n i
-- runNet !n !i = trace "runNet" $ last $ runNet' n i

-- Collects all intermediate activations into a list of activations
runNet' :: Network -> Input -> [Activation]
runNet' []        !i = [i]
runNet' (w : net) !i = i : runNet' net α where α = activation w i
-- runNet' (w : net) !i = trace "runNet'" $ i : runNet' net α where α = activation w i

-- Runs the neural network over a list of inputs and collects every "answer"
collectOutputs :: Network -> [Input] -> [Output]
collectOutputs net = fmap (runNet net)

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

logistic' :: Floating a => a -> a
logistic' x = o * (1 - o) where o = logistic x

activation :: Weights -> Input -> Activation
activation !w !v = logistic $ runLayer w v

-- (Not used when actually running the neural network. The slides didn't
-- mention using a loss function at all to update the weights...)
-- Target -> Output -> Error
loss :: Target -> Output -> Double
loss = (* 0.5) ... LA.sumElements ... (^ 2) ... (-)

-- Not used anywhere in the codebase
lossO :: Floating a => a -> a -> a -> a
lossO t y α = logistic' y * (α - t)

-- If there's an error in this code, it is likely to be in the delta functions
-- or the weight updates
δo :: Target -> Output -> Vector Double
δo !t !α = α * (1 - α) * (t - α)
-- δo !t !α = trace "δo" $ α * (1 - α) * (t - α)

δh :: Weights -> Activation -> Vector Double -> Vector Double
δh (W !wB !wN) !h !δ' = h * (1 - h) * (LA.tr' wN #> δ')
-- δh (W !wB !wN) !h !δ' = trace "δh" $ h * (1 - h) * (LA.tr' wN #> δ')
-- Can't be this because it's not an inconsistent matrix error...

-- The error terms; algorithm for this and the deltas given by the slides
errors :: Network -> Target -> [Input] -> [Vector Double]
errors [!ih, !ho] !t [!αH, !αO] = [δhidden, δoutput]
-- errors [!ih, !ho] !t [!αH, !αO] = trace "errors" [δhidden, δoutput]
 where
  δoutput = δo t αO
  δhidden = δh ho αH δoutput

-- Weight deltas with momentum (only the deltas, not the actual change)
-- scalar is a function that allows me to multiply a matrix (or vector) by a single number
-- δ and i need to be the same size. δ should be 20
-- if wNΔ is (n,m) size, LA.asRow δ*i needs to be (x,m), LA.asColumn δ*i needs to be (n,x)
weightΔ :: Vector Double -> Input -> Prev (Diff Weights) -> (Weights, Diff Weights)
weightΔ !δ !i w@(W !wBΔ !wNΔ) = (w, W wΔb wΔn)
-- weightΔ !δ !i w@(W !wBΔ !wNΔ) = trace "weightΔ" (w, W wΔb wΔn)
 where
  wΔb = LA.scalar η * δ + LA.scalar α * wBΔ
  wΔn = LA.asColumn (LA.scalar η * δ * i) + LA.scalar α * wNΔ
  -- tracePrint = "δ: "      ++ show (LA.size δ)
  --           ++ " i: "     ++ show (LA.size i)
  --           ++ " wNΔ: "   ++ show (LA.size wNΔ)
  --           ++ " wBΔ: "   ++ show (LA.size wBΔ)
            -- ++ " asCol: " ++ show (LA.size $ LA.scalar η * δ * i)
            -- δ: 20 i: 784 wNΔ: (20,784) wBΔ: 20
            -- passing in wrong δ?

wgtUpd :: Weights -> Diff Weights -> Weights
wgtUpd (W wB wN) (W wB' wN') = W (wB + wB') (wN + wN')
-- wgtUpd (W wB wN) (W wB' wN') = trace "wgtUpd" $ W (wB + wB') (wN + wN')

backprop -- Given the list of prev diff weights for every layer as well
  :: (Prev Network, Network)
  -> Target
  -> Input
  -> [Prev (Diff Weights)]
  -> ([Diff Weights], Network)
backprop (net', net) !target !input w' = (wΔs, zipWith wgtUpd net wΔs)
-- backprop (net', net) !target !input w' = trace "Backprop" (wΔs, zipWith wgtUpd net wΔs)
 where
  αs  = tail $ runNet' net input -- the activations, excluding the input vector
  δs  = errors net target αs     -- length 2 lst
  wΔs = snd <$> zipWith3 weightΔ δs αs w'

-- Iterate through a list of inputs (and list of targets) and successively
-- train on those inputs.
epoch
  :: (Prev Network, Network)   -- Old network (zeros in first run), current network
  -> [Target]                  -- List of input vectors
  -> [Input]                   -- List of target vectors
  -> [Diff Weights]
  -> ([Diff Weights], Network) -- Resulting network
epoch (_    , net ) []       []       wΔs   = (wΔs, net)
epoch (!net', !net) (t : ts) (i : is) !wΔs' = epoch (net, next) ts is wΔs
-- epoch (!net', !net) (t : ts) (i : is) !wΔs' = trace "Epoch happening" $ epoch (net, next) ts is wΔs
  where (wΔs, next) = backprop (net', net) t i wΔs'

train
  :: Int -- Epochs to train for
  -> (Prev Network, Network) -- (Rest of params are identical to epoch)
  -> ([Target], [Target])
  -> ([Input], [Input])
  -> [Diff Weights]
  -> IO ()
train 0 nets (tstT, t) (tstI, i) wΔs' = do
  let final    = snd $ epoch nets t i wΔs'
      outputs  = collectOutputs final i
      oTest    = collectOutputs final tstI
      accuracy = totalError (t, outputs)
      tstAcc   = totalError (tstT, oTest)
  print $ "Epoch 50 accuracy: " ++ show accuracy
  print $ "Epoch 50 test accuracy: " ++ show tstAcc
  -- print out stuff
train n (net', net) t'@(tstT, t) i'@(tstI, i) wΔs' = do
  let (wΔs, next) = epoch (net', net) t i wΔs'
      outputs     = collectOutputs next i
      oTest       = collectOutputs next tstI
      accuracy    = totalError (t, outputs)
      tstAcc      = totalError (tstT, oTest)
  print $ "Epoch " ++ show (50-n) ++ " accuracy: "      ++ show accuracy
  print $ "Epoch " ++ show (50-n) ++ " test accuracy: " ++ show tstAcc
  train (n - 1) (net, next) t' i' wΔs

-- Simple debugging function that tells me the shape of a particular network.
-- Useful for ensuring my matrices are the right shape
sizeOf :: Network -> [(Int, (Int, Int))]
sizeOf [W hB hN, W oB oN] = [(LA.size hB, LA.size hN), (LA.size oB, LA.size oN)]

totalError :: ([Target], [Output]) -> Double
totalError (!ts, !os) = sum (zipWith loss ts os) / fromIntegral (length ts)

main :: IO ()
main = do
  -- Yes, this is slightly ugly...
  !file <- fromMaybe (error "file decoding failed")
    <$> decodeIDXFile "/home/jaredweakly/Documents/Classes/CS445/train-images-idx3-ubyte"
  !labels <- fromMaybe (error "label decoding failed") <$> decodeIDXLabelsFile
    "/home/jaredweakly/Documents/Classes/CS445/train-labels-idx1-ubyte"
  !testFile <- fromMaybe (error "test file decoding failed")
    <$> decodeIDXFile "/home/jaredweakly/Documents/Classes/CS445/t10k-images-idx3-ubyte"
  !testLabels <- fromMaybe (error "test label decoding failed") <$> decodeIDXLabelsFile
    "/home/jaredweakly/Documents/Classes/CS445/t10k-labels-idx1-ubyte"

  -- mnist : 60_000 [(target, Vector 784 Double)]
  let !mnist = fromMaybe (error "labeling failed") $ labeledDoubleData labels file
      !mnist' =
        fromMaybe (error "labeling failed") $ labeledDoubleData testLabels testFile
      !initN20   = runIdentity $ zeroNet 784 [20] 10
      !initN50   = runIdentity $ zeroNet 784 [50] 10
      !initN100  = runIdentity $ zeroNet 784 [100] 10
      !testset = mnist' ++ mnist

  print "Loaded up data"
  -- inputs : (60000><785) Matrix Double
  -- targets : Vector 10 Double. 0.1 everywhere but target which is 0.9
  let mkSparseVectors [] = [] :: [Vector Double]
      mkSparseVectors (x : xs) =
        LA.assoc 10 0.1 [(x, 0.9)] : mkSparseVectors xs :: [Vector Double]

      normalize :: V.Vector Double -> Vector Double
      normalize            = (/ 255) . V.convert

      !(!targets, !inputs) = bimap mkSparseVectors id . unzip $ normalize <$$> mnist
      !(!tstT   , !tstI  ) = bimap mkSparseVectors id . unzip $ normalize <$$> testset

  !weights20  <- randNet 784 [20] 10
  !weights50  <- randNet 784 [50] 10
  !weights100 <- randNet 784 [100] 10

  print "Beginning training with n=20"
  train 50 (initN20, weights20) (tstT, targets) (tstI, inputs) initN20

  print "Beginning training with n=50"
  train 50 (initN50, weights50) (tstT, targets) (tstI, inputs) initN50

  print "Beginning training with n=100"
  train 100 (initN100, weights100) (tstT, targets) (tstI, inputs) initN100





-- Helper functions I was too lazy to import
-- These are not essential to understanding any of the actual code.

-- Lets me write pointfree code using functions that takes two arguments
-- instead of just one
infixr 9 ...
(...) :: (b -> c) -> (a1 -> a2 -> b) -> a1 -> a2 -> c
(...) = (.) . (.)

-- Lets me map over something inside another structure.
-- eg: (+1) <$$> [[a]] would map over all of the inner lists
infixl 4 <$$>
(<$$>) :: (Functor f2, Functor f1) => (a -> b) -> f1 (f2 a) -> f1 (f2 b)
(<$$>) = fmap fmap fmap

both :: (a -> b) -> (a, a) -> (b, b)
both f (a, a') = (f a, f a')
