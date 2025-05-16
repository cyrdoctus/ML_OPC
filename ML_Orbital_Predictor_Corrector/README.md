# Awesome Project Name
[![Version](https://img.shields.io/badge/Version-0.0.9-blue.svg)](https://github.com/your-organization/your-repo/releases/tag/v1.0.0)
[![Last Updated](https://img.shields.io/badge/Last%20Update-May%2011,%202025-yellow.svg)](https://github.com/your-organization/your-repo/commits/main)
[![Organization](https://img.shields.io/badge/Organization-University_of_Arizona-lightgrey.svg)](https://www.arizona.edu/admissions?utm_source=google&utm_medium=cpc&utm_term=university%20of%20arizona&utm_campaign=brand_az_ca_tx_search&gad_source=1&gad_campaignid=17563208285&gbraid=0AAAAAovfQnip6Rb572lB_MfOjEvA_U2qN&gclid=Cj0KCQjwlYHBBhD9ARIsALRu09qtSSV6MySsCPyh89Veb1w1LlkTNXLqh1hnsjAx9ve-qIC2J9w-rnMaAprlEALw_wcB)
[![Course](https://img.shields.io/badge/Course-ECE_579-orange.svg)](https://infosci.arizona.edu/course/ece-579-principles-artificial-intelligence)

## Author

**Adam Nekolny**

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- 
## Description

ML Orbital Predictor Corrector project focuses on training and applying a dual set of machine leanrning algorithms. In the current day and age of spacecraft it might be possible to apply AI for protecting spacecraft autonomosly as
form of rapid response and self correction. While typical onboard system of small satelites use low powered modules ot ocnserve energy the current progress in AI and compute may alow some highly efficient systems to run AI on board.
Here is where this project comes, typically a predictor code would run an extensive numerical simulation that would propagate the spacecraft forward and based on that data assume if the corrector code
need to be applied. This can be very computationally demanding as such there is a potential to have less accurate but still valid AI that predicts the satelites position
around an hour forward and based on the verification code determines if that position is good or bad. If its marked as too low or two high compared to the desired orbit, the corrector code is to be applied.
In case the corrector AI quickly predicts the needed Delta-V for moving as close to the target orbit as possible, stabilizing the spacecraft. The objective of this project is to trian these AI's and compare how accruate are they,
if it can be a viable option, as well as how much computationally demanding it is to run the AI compared to just the numerical integration.

NOTE: The currwent version is incomplete as due to the limitation of computational resources and time, the AI could not me trained as such while the backend of AI code is within the files above the model is not trained, and needs to be adjusted. Curently the coede is forced for CPU training which was result on noncompatibility of Pytorch version and my current GPU. This is planned to be resolved in future versions by me rending server class gpus and training AI on them.

## Installation

This segment is currently in progress as the code is being developed.

