"""
This script allows training of any of the deep learning models presented in the paper 'Are Deep Learning
Methods Suitable for Downscaling Global Climate Projections? Review and Comparison of Existing Models,' as
well as computing projections for any Global Climate Model (GCM). Due to space limitations, we demonstrate
how to generate these projections with a limited amount of data. Adapting this script to other datasets
should be straightforward.
"""

import json
import xarray as xr
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import src.trans as trans
import src.deep.utils as deep_utils
import src.deep.models as deep_models
import src.deep.loss as deep_loss
import src.deep.train as deep_train
import src.deep.pred as deep_pred


"""
The file ./configs/path.json defines all the paths for loading and saving data and models. By default, it
contains relative paths, so the script should work as-is.
"""

paths = json.load(open('./configs/paths.json'))
data_path = paths['data']
model_path = paths['models']
asym_path = paths['asym']
preds_path = paths['data_preds']
gcm_raw = paths['gcm_raw']
proj_path = paths['data_proj']


"""
If a GPU is detected, the models are executed on it for both training and inference. Note that for GPU
execution it is necessary to install the proper Pytorch version.
"""

device = ('cuda' if torch.cuda.is_available() else 'cpu')


"""
These variables control the deep learning model to be trained:
    - var_target: Variable to downscale (tasmin, tasmax or pr)
    - dl_architecture: Architecture of the deep learning model (deepesd or unet)
    - loss: Loss function to minimize (MSE, SQR, ASYM or STO)

As demonstration, we will train a model to downscale the minimum temperature (tasmin) 
with the DeepESD architecture (deepesd) by minimizing the Mean Square Error (MSE)
"""

var_target = 'tasmin'
dl_architecture = 'deepesd'
loss = 'MSE'


"""
It is not possible to train a model to minimize the ASYM loss function when downscaling a variable other
than precipitation, so we check for this constraint.
"""

if (loss == 'ASYM') and (var_target != 'pr'):
    raise ValueError('ASYM loss functions is only compatible with precipitation')


"""
As previously mentioned, this repository includes small datasets to demonstrate how to run the code.
Specifically, these are subsets of the datasets used to compute the results shown in the manuscript.
Modify these lines as needed, as there are two xarray.Dataset objects corresponding to the predictor
and predictand. Additionally, it is important that the spatial coordinates use 'lat' and 'lon' as
names, rather than the longer forms 'latitude' and 'longitude.'
"""

# Load predictor
predictor_train_filename = f'{data_path}/predictor_train.nc'
predictor_test_filename = f'{data_path}/predictor_test.nc'

predictor_train = xr.open_dataset(predictor_train_filename).load()
predictor_test = xr.open_dataset(predictor_test_filename).load()
predictor = xr.merge([predictor_train, predictor_test])

# Load predictand
# We control what predictand to load based on var_target
predictand_train_filename = f'{data_path}/predictand_{var_target}_train.nc'
predictand_test_filename = f'{data_path}/predictand_{var_target}_test.nc'

predictand_train = xr.open_dataset(predictand_train_filename).load()
predictand_test = xr.open_dataset(predictand_test_filename).load()
predictand = xr.merge([predictand_train, predictand_test])


"""
Remove days with NaNs in the predictor, align the predictor and predictand in
time, and split the data into training and test periods.
"""

predictor = trans.remove_days_with_nans(predictor)
predictor, predictand = trans.align_datasets(predictor, predictand, 'time')

period_train = ('01-01-2010', '06-01-2010')
period_test = ('07-01-2010', '12-31-2010')

x_train = predictor.sel(time=slice(*period_train))
y_train = predictand.sel(time=slice(*period_train))

x_test = predictor.sel(time=slice(*period_test))
y_test = predictand.sel(time=slice(*period_test))


"""
Standardize the predictors.
"""

x_train_stand = trans.standardize(data_ref=x_train, data=x_train)


"""
To handle NaNs in the predictand, we compute a mask to define the grid points to model. For the DeepESD
model, due to its final dense layer, we remove all grid points containing NaNs. For the U-Net, since all
points will be computed (although only those corresponding to non-NaNs are evaluated), we do not apply
the mask.
"""

# Compute a mask of non-NaN values. This is required to reshape the deep learning model's prediction
# into a valid format
y_mask = trans.compute_valid_mask(y_train) 

# Stack in one dimension (gridpoint)
y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

# Remove NaNs following y_mask. This is useful for models with a
# final fully-connected layer
y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)

# For the U-Net, all values are predicted (including NaNs), but the loss
# function is only evaluated for the non-NaN values. Therefore, there is
# no need for masking, allowing the model to handle grid points that mix
# NaN and non-NaN values
if dl_architecture == 'deepesd':
    y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                             drop=True) # Filter y_train w.r.t. y_mask
elif dl_architecture == 'unet':
    y_train_stack_filt = y_train_stack.copy(deep=True)

"""
Preprocess the predictand for the SQR and STO (for precipitation) loss functions. 
"""

if loss == 'SQR':
    y_train_stack_filt = y_train_stack_filt ** (1/2)

# To accurately model the Gamma distribution for wet days, we
# subtract the threshold_pr from the predictand, so the amount
# corresponding to dry days is not considered
if (var_target == 'pr') and (loss == 'STO'):
    threshold_pr = 0.1
    y_train_stack_filt = deep_utils.precipitation_NLL_trans(data=y_train_stack_filt,
                                                            threshold=threshold_pr)


"""
Compute the gamma distributions for the ASYM loss function. These only need to be computed once, as they
will be loaded if they already exist.
"""

if (var_target == 'pr') and (loss == 'ASYM'):
    loss_function = deep_loss.Asym(ignore_nans=True,
                                   asym_path=asym_path)

    if loss_function.parameters_exist():
        loss_function.load_parameters()
    else:
    # It is important to always compute the ASYM parameters using the full
    # predictand domain (including NaNs) to avoid shape issues
    # when computing the loss function during model training
        loss_function.compute_parameters(data=y_train_stack,
                                         var_target=var_target)

# If working with the DeepESD model, filter the computed gamma
# gamma distributions to match the predictand
if (loss == 'ASYM') and (dl_architecture == 'deepesd'):
    loss_function.mask_parameters(mask=y_mask)


"""
Create the training and validation DataLoaders by subsetting 90% and 10% of the training data, respectively.
"""

# Convert data from xarray to numpy
x_train_stand_arr = trans.xarray_to_numpy(x_train_stand)
y_train_arr = trans.xarray_to_numpy(y_train_stack_filt)

# Create Dataset
train_dataset = deep_utils.StandardDataset(x=x_train_stand_arr,
                                           y=y_train_arr)

# Split into training and validation sets
train_dataset, valid_dataset = random_split(train_dataset,
                                            [0.9, 0.1])

# Create DataLoaders
batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=True)


"""
Load the deep learning model based on var_target, dl_architecture and loss variables.
"""

model_name = f'{dl_architecture}_{loss}_{var_target}'

if loss == 'STO':
    stochastic = True
else:
    stochastic = False

if var_target in ('tasmin', 'tasmax'):
    if dl_architecture == 'deepesd':
        model = deep_models.DeepESDtas(x_shape=x_train_stand_arr.shape,
                                       y_shape=y_train_arr.shape,
                                       filters_last_conv=10,
                                       stochastic=stochastic)
    elif dl_architecture == 'unet':
        model = deep_models.UnetTas(x_shape=x_train_stand_arr.shape,
                                    y_shape=y_train_arr.shape,
                                    stochastic=stochastic,
                                    input_padding=(11, 11, 16, 17),
                                    kernel_size=3, padding='same',
                                    batch_norm=False, trans_conv=True)

elif var_target == 'pr':
    if dl_architecture == 'deepesd':
        model = deep_models.DeepESDpr(x_shape=x_train_stand_arr.shape,
                                      y_shape=y_train_arr.shape,
                                      filters_last_conv=1,
                                      stochastic=stochastic)
    elif dl_architecture == 'unet':
        model = deep_models.UnetPr(x_shape=x_train_stand_arr.shape,
                                   y_shape=y_train_arr.shape,
                                   stochastic=stochastic,
                                   input_padding=(11, 11, 16, 17),
                                   kernel_size=3, padding='same',
                                   batch_norm=False, trans_conv=True)


"""
Set the training hyperparameters.
"""

num_epochs = 10000
patience_early_stopping = 60

learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)


"""
Load the loss function to minimize.
"""

if stochastic:
    if var_target in ('tasmin', 'tasmax'):
        loss_function =  deep_loss.NLLGaussianLoss(ignore_nans=True)
    elif var_target == 'pr':
        loss_function = deep_loss.NLLBerGammaLoss(ignore_nans=True)
else:
    if loss in ('MSE', 'SQR'):
        loss_function = deep_loss.MseLoss(ignore_nans=True)
    elif loss == 'ASYM':
        pass # It has been already defined

# Move the parameters of the ASYM loss function to the device
if loss == 'ASYM':
    loss_function.prepare_parameters(device=device)


"""
Train the deep learning model. The model will be automatically saved to model_path.
"""

train_loss, val_loss = deep_train.standard_training_loop(model=model, model_name=model_name, model_path=model_path,
                                                         device=device, num_epochs=num_epochs,
                                                         loss_function=loss_function, optimizer=optimizer,
                                                         train_data=train_dataloader, valid_data=valid_dataloader,
                                                         patience_early_stopping=patience_early_stopping)


"""
Load the weights of the model saved during the training.
"""

model.load_state_dict(torch.load(f'{model_path}/{model_name}.pt'))

"""
Compute the predictions on the test set. To do so, first test data is standardized using the training set as
a reference. To avoid any out-of-memory (OOM) errors, predictions are computed in batches of 16. The prediction
is saved in preds_path.
"""

x_test_stand = trans.standardize(data_ref=x_train, data=x_test)

if stochastic:
    if var_target in ('tasmin', 'tasmax'):
        pred_test = deep_pred.compute_preds_gaussian(x_data=x_test_stand, model=model,
                                                     device=device, var_target=var_target,
                                                     mask=y_mask, batch_size=16)
    elif var_target == 'pr':
        # This function handles the preprocessing of precipitation data 
        # when minimizing the STO loss function.
        pred_test = deep_pred.compute_preds_ber_gamma(x_data=x_test_stand, model=model,
                                                      threshold=threshold_pr, device=device,
                                                      var_target=var_target, mask=y_mask,
                                                      batch_size=16)
else:
    pred_test = deep_pred.compute_preds_standard(x_data=x_test_stand, model=model,
                                                 device=device, var_target=var_target,
                                                 mask=y_mask, batch_size=16)

# Handle the preprocessing when minimizing the
# STO loss function.
if loss == 'SQR':
    pred_test = pred_test ** 2

file_name = f'{model_name}_pred_test'
pred_test.to_netcdf(f'{preds_path}/{file_name}.nc')


"""
Load the historical and future predictors from the GCM to perform downscaling. To demonstrate how this code works,
we use a small subset of one of the GCMs downscaled in the manuscript as an example. The corresponding xr.Dataset
should follow the same structure as the predictor.
"""

gcm_hist_filename = f'{gcm_raw}/gcm_predictors_historical.nc'
gcm_hist = xr.open_dataset(f'{gcm_raw}/gcm_predictors_historical.nc').load()

gcm_fut_filename = f'{gcm_raw}/gcm_predictors_future.nc'
gcm_fut = xr.open_dataset(f'{gcm_raw}/gcm_predictors_future.nc').load()


"""
Before feeding the data to the model, and as explained in the manuscript, GCM predictors are first bias-corrected
and then standardized.
"""

gcm_hist_corrected = trans.scaling_delta_correction(data=gcm_hist,
                                                    gcm_hist=gcm_hist, obs_hist=x_train)
gcm_fut_corrected = trans.scaling_delta_correction(data=gcm_fut,
                                                   gcm_hist=gcm_hist, obs_hist=x_train)

gcm_hist_corrected_stand = trans.standardize(data_ref=x_train, data=gcm_hist_corrected)
gcm_fut_corrected_stand = trans.standardize(data_ref=x_train, data=gcm_fut_corrected)


"""
Compute the projections for the historical and future periods in a manner similar to the predictions for the test
set, and save them to the file_name_hist and file_name_fut paths.
"""

if stochastic:
    if var_target in ('tasmin', 'tasmax'):
        proj_historical = deep_pred.compute_preds_gaussian(x_data=gcm_hist_corrected_stand, model=model,
                                                           device=device, var_target=var_target, mask=y_mask,
                                                           batch_size=16)
        proj_future = deep_pred.compute_preds_gaussian(x_data=gcm_fut_corrected_stand, model=model,
                                                       device=device, var_target=var_target, mask=y_mask,
                                                       batch_size=16)
    elif var_target == 'pr':
        proj_historical = deep_pred.compute_preds_ber_gamma(x_data=gcm_hist_corrected_stand, model=model,
                                                            threshold=threshold_pr, device=device,
                                                            var_target=var_target, mask=y_mask,
                                                            batch_size=16)
        proj_future = deep_pred.compute_preds_ber_gamma(x_data=gcm_fut_corrected_stand, model=model,
                                                        threshold=threshold_pr, device=device,
                                                        var_target=var_target, mask=y_mask,
                                                        batch_size=16)
else:
    proj_historical = deep_pred.compute_preds_standard(x_data=gcm_hist_corrected_stand, model=model,
                                                       device=device, var_target=var_target,
                                                       mask=y_mask, batch_size=16)
    proj_future = deep_pred.compute_preds_standard(x_data=gcm_fut_corrected_stand, model=model,
                                                   device=device, var_target=var_target,
                                                   mask=y_mask, batch_size=16)

if loss == 'SQR':
    proj_historical = proj_historical ** 2
    proj_future = proj_future ** 2

file_name_hist = f'GCM_proj_historical_{model_name}'
file_name_fut = f'GCM_proj_future_{model_name}'

proj_historical.to_netcdf(f'{proj_path}/{file_name_hist}.nc')
proj_future.to_netcdf(f'{proj_path}/{file_name_fut}.nc')
