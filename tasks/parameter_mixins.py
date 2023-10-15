import luigi
import law


class DatasetDependency(object):
    dataset_version = luigi.Parameter(
        default="dataset_version_01", description="Version Tag for dataset to save file with"
    )

    compression = luigi.BoolParameter()

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.dataset_version,)
        return parts


class TrainingDependency(object):
    training_version = luigi.Parameter(
        default="training_version_01", description="Version Tag for training to save file with"
    )
    epochs = luigi.IntParameter(default=30)
    FP16 = luigi.BoolParameter()
    compiled = luigi.BoolParameter()
    adv = luigi.BoolParameter()
    model = luigi.ChoiceParameter(choices=['DeepJet','DeepJetTransformer','ParticleTransformer','ParticleTransformerBig','ParticleTransformerHuge',
                                           'BetterParticleTransformer','ParticleRetention'], var_type=str)
    scheduling = luigi.ChoiceParameter(choices=['none','epoch_lin_decay','batch_lin_decay','batch_cosine_warmup'], var_type=str)

    def store_parts(self):
        parts = super().store_parts()
        parts += (self.training_version,)
        parts += ("epochs_{0:d}".format(self.epochs),)

        return parts
