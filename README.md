[![MIT License](https://img.shields.io/badge/License-MIT-lightgray.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/Python-3.8.5-blue.svg)
<!-- Add in additional badges as appropriate -->

![Banner of NHS AI Lab Skunkworks ](docs/banner.png)

# NHS AI Lab Skunkworks project: C359 - NHS Ambulance Handover Delay Predictor

> A pilot project for the NHS AI (Artificial Intelligence) Lab Skunkworks team, C359 - NHS Ambulance Handover Delay Predictor, will use statistical analysis and machine learning to understand whether AI approaches can be used to create a proactive response to the redirect of ambulances between hospitals on a given day in order to minimise the time spent waiting for patient handover and thereby maximise the time available to respond to patient calls for help
C359 - NHS Ambulance Handover Delay Predictorwas selected as a project in Q2 2022 following a succesful pitch to the AI Skunkworks problem-sourcing programme.

## Intended Use

This proof of concept ([TRL 4](https://en.wikipedia.org/wiki/Technology_readiness_level)) is intended to demonstrate the technical validity of applying X technique to Y dataset in order to solve Z. It is not intended for deployment in a clinical or non-clinical setting without further development and compliance with the [UK Medical Device Regulations 2002](https://www.legislation.gov.uk/uksi/2002/618/contents/made) where the product qualifies as a medical device.

## Data Protection

This project was subject to a Data Protection Impact Assessment (DPIA), ensuring the protection of the data used in line with the [UK Data Protection Act 2018](https://www.legislation.gov.uk/ukpga/2018/12/contents/enacted) and [UK GDPR](https://ico.org.uk/for-organisations/dp-at-the-end-of-the-transition-period/data-protection-and-the-eu-in-detail/the-uk-gdpr/). No data or trained models are shared in this repository.

## Background

South Central Ambulance Service (SCAS) has around 2.5 million total patient contacts per year across the services of Hampshire, Thames Valley, Surrey and Sussex.  To ensure the delivery of an effective and efficient service, SCAS need to make informed decisions, drawing upon the best possible management information available.
The decision on where to take ambulance patients is complex taking in factors such as severity of illness, urgency of the situation, geography, travel time and handover/queuing time.  With national targets on handover times and NHS trusts publishing their ability to meet these targets, there is an opportunity to determine if the large volume of electronic data available from patients who have been taken to hospital can be used within Artificial intelligence (AI)/Machine Learning (ML) to predict where patients should be sent to minimise waiting times.  Such techniques are commonly used across industries that provide services or sell goods through, but not limited to: optimising staffing schedules, reducing waiting times (e.g. to answer calls or to be physically seen) and increasing the robustness of a queuing system to the inevitable variation in demand for a service.  
Looking across an NHS trust, knowing where ambulances can handover patients to inform balancing for care outcomes could potentially improve the performance and safety for both the ambulance trusts and hospitals, whilst also: 
•	reducing the stress and improve the overall experience for the patient, 
•	reducing the overall clinical risk as handover from the ambulance to the hospital will happen as quickly as possible
•	reducing operational pressures on the ambulance service provider by reducing the amount of reactive management, thereby also reducing staffing stress levels
•	increasing both the hospital and ambulance efficiency – for every patient waiting to be admitted to the hospital, there is an ambulance crew that is not able to attend another call


## Model selection

_Include a high-level rationale of the approach you took._

## Known limitations

_Include known limitations and issues with your approach._

## Data pipeline

_Include an ideally visual representation of data flow, and include a link to a data dictionary/data requirements._

## Getting Started

1. Clone this repository: `git clone git@github.com:nhsx/skunkworks-template.git`
2. Switch to the `develop` branch: `git branch develop`
3. Create a new virtual environment: `mkvirtualenv my-project` or `pyenv virtualenv 3.8.5 my-project`
4. Install the requirements of this template: `pip install -r requirements.txt`
5. Enable pre-commits: `pre-commit install`

_Include a brief overview of the codebase. Include setup and execution instructions, including any required environment variables. Consider using a `virtual environment` for dependency management._

## NHS AI Lab Skunkworks
The project is supported by the NHS AI Lab Skunkworks, which exists within the NHS AI Lab at NHSX to support the health and care community to rapidly progress ideas from the conceptual stage to a proof of concept.

Find out more about the [NHS AI Lab Skunkworks](https://www.nhsx.nhs.uk/ai-lab/ai-lab-programmes/skunkworks/).
Join our [Virtual Hub](https://future.nhs.uk/connect.ti/system/text/register) to hear more about future problem-sourcing event opportunities.
Get in touch with the Skunkworks team at [aiskunkworks@nhsx.nhs.uk](aiskunkworks@nhsx.nhs.uk).


## Licence

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
