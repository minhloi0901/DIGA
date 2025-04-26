from .ttda_module import DomainOptimalTransport

class DIGA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.model.num_classes
        
        # Initialize backbone and classifier
        self.backbone = ResNetMulti(cfg.model.backbone)
        self.classifier = Classifier_Module(cfg.model.classifier)
        
        # Initialize DOT module
        self.dot = DomainOptimalTransport(num_features=2048)  # Assuming ResNet-101 backbone
        
        # Initialize other components
        self.bn_lambda = cfg.model.bn_lambda
        self.proto_lambda = cfg.model.proto_lambda
        self.fusion_lambda = cfg.model.fusion_lambda
        self.confidence_threshold = cfg.model.confidence_threshold
        self.proto_rho = cfg.model.proto_rho
        self.prior_ = cfg.model.prior_
        
    def forward(self, x, is_source=True):
        # Extract features
        features = self.backbone(x)
        
        if is_source:
            # Update source statistics
            self.dot.update_source_stats(features)
        else:
            # Update target statistics and compute transport plan
            self.dot.update_target_stats(features)
            self.dot.compute_transport_plan()
            
            # Apply domain optimal transport
            features = self.dot(features)
        
        # Get predictions
        pred = self.classifier(features)
        
        return pred, features
        
    def get_parameters(self):
        params = []
        params.extend(self.backbone.parameters())
        params.extend(self.classifier.parameters())
        params.extend(self.dot.parameters())
        return params 