import torch
import pytest
import models_02 as models

@pytest.fixture
def input_batch():
  batch_size = 4

  X = torch.rand(batch_size, 2, 512, 512)

  y = torch.randint(0, 2, (batch_size,))
  return X, y

@pytest.fixture
def input_batch_three():
  batch_size = 4

  X = torch.rand(batch_size, 3, 512, 512)

  y = torch.randint(0, 2, (batch_size,))
  return X, y

def test_brainnet_v1(input_batch):
  X, y = input_batch
  batch, channels, heigh, weight = X.shape

  model = models.BrainNetv1()
  try:
      output = model(X)
  except Exception as e:
      pytest.fail(f"Forward method raised an exception: {e}")

  assert output.shape == torch.Size([batch, 3])

  assert isinstance(output, torch.Tensor)

def test_brainnet_v2(input_batch):
  X, y = input_batch
  batch, channels, heigh, weight = X.shape

  model = models.BrainNetv2()
  try:
      output = model(X)
  except Exception as e:
      pytest.fail(f"Forward method raised an exception: {e}")

  assert output.shape == torch.Size([batch, 3])

  assert isinstance(output, torch.Tensor)

def test_brainnet_v3(input_batch):
  X, y = input_batch
  batch, channels, heigh, weight = X.shape

  model = models.BrainNetv3()
  try:
      output = model(X)
  except Exception as e:
      pytest.fail(f"Forward method raised an exception: {e}")

  assert output.shape == torch.Size([batch, 3])

  assert isinstance(output, torch.Tensor)

def test_brainnet_v4(input_batch):
  X, y = input_batch
  batch, channels, heigh, weight = X.shape

  model = models.BrainNetv4()
  try:
      output = model(X)
  except Exception as e:
      pytest.fail(f"Forward method raised an exception: {e}")

  assert output.shape == torch.Size([batch, 3])

  assert isinstance(output, torch.Tensor)

def test_brainmobilenet(input_batch_three):
  X, y = input_batch_three
  batch, channels, heigh, weight = X.shape

  model = models.BrainGoogleNet()
  try:
      output = model(X)
  except Exception as e:
      pytest.fail(f"Forward method raised an exception: {e}")

  assert output.shape == torch.Size([batch, 3])

  assert isinstance(output, torch.Tensor)