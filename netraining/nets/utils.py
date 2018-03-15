class MultiConvClassifier(nn.Module):
    def __init__(self, in_features, class_nbs):
        super().__init__()
        self.in_features = in_features
        self.class_nbs = class_nbs
        classifier_names = []
        for classifier_idx, class_nb in enumerate(class_nbs):
            classifier = Unit3Dpy(
                in_channels=in_features,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                activation=None,
                use_bias=True,
                use_bn=False)

            classifier_name = 'classifier_{}'.format(classifier_idx)
            classifier_names.append(classifier_name)
            self.add_module(classifier_name, classifier)
        self.classifier_names = classifier_names

    def forward(self, inp):
        outputs = []
        for classifier_name in self.classifier_names:
            classifier = getattr(self, classifier_name)
            out = classifier(inp)
            outputs.append(out)
        return outputs
