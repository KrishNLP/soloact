# SOLO ACT

One Paragraph of project description goes here




Not all deep networks need to obfuscate learning.


Music has often thrived on transforming faults into influential sound effects. Before professional studio production enabled granular tweaks in sound, standalone guitar effects emerged from deliberately converting hardware faults—often caused by the limitations of amplifiers—into positive features.

# DATA
=======

## Foundations

"The IDMT-SMT-GUITAR database is a large database for automatic guitar transcription. Seven different guitars in standard tuning were used with varying pick-up settings and different string measures to ensure a sufficient diversification in the field of electric and acoustic guitars..."

"The dataset consists of four subsets. The first (*SOLOACT*: what we use) contains all introduced playing techniques (plucking styles: finger-style, muted, picked; expression styles: normal, bending, slide, vibrato, harmonics, dead-notes) and is provided with a bit depth of 24 Bit."

We don't venture into the available polyphonic data. Instead we mix monophonic pitch classes into 3 commonly utilized chord structures; triads, powerchords and septachords at strict 44100 Hz sampling rates.

### Powerchords

- https://www.youtube.com/watch?v=ik9TGv4YNX8

## Engineering

We programmatically mix tracks with combinations of overdrive, reverberance, chorus, phaser - feeling these best capture hallmark sounds of electric guitar play. Some effects permitted randomization between on/off states while others i.e. overdrive were too audibly fundamental for this kind of intermittence. To capture granularity, and realistic waveforms, we impose strict bounds on each effect with values transcribed by headphones and hand. The following image depicts a clear response of powerchord waveforms to multi-effect augmentations.

![alt text](assets/multi_class_effect.png)

https://www.youtube.com/watch?v=MpHA8hoc9SU

## Feature extraction

#### MFCC







## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## TUTORIAL

**Draft**

## CLI
======

All major features are operable from root's main.py accessible with conventional argparse notation

```
python main.py -h
```

![alt text](assets/cli_intro_1.png)

##### 1. PARSE ANNOTATIONS
======

The IDMT-SMT-Guitar dataset contains track-wise xml annotations.
These are necessary for labelling training data and anchoring our name-based dependencies.

```
python main.py make_meta -r
```

##### 2. GENERATE CHORDS
======

Choose between triads, powerchords and septchords
You'll find how strategies are composed in "soloact/soloact/data"

![alt text](assets/write_chords.png)

##### 3. Augment your data
======

Since this is a self contained study we offer feature extraction as means of reducing storage required for training data.
Using the -wa argument will write augmented tracks to a nested directory in the "data/interim/" folder. 
Modeling your data for classification will induce randomized on/off states at the effect level for those not required to persist - see "soloact/models" for .yaml configs used in the subsequent function  - train existing.



![alt text](assets/augment_data.png)


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

### Glossary

- https://en.wikipedia.org/wiki/Pitch_class
- https://en.wikipedia.org/wiki/44,100100_Hz

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
