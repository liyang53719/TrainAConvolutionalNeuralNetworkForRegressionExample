classdef RegressionMAELayer_ly < nnet.layer.RegressionLayer
               
    methods
        function layer = RegressionMAELayer_ly(name)
            % Create an exampleRegressionMAELayer

            % Set layer name
            if nargin == 1
                layer.Name = name;
            end

            % Set layer description
            layer.Description = 'Regression Layer with MAE loss by ly';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Returns the MAE loss between the predictions Y and the
            % training targets T

            % Calculate MAE
            K = size(Y,3);
            meanAbsoluteError = sum(abs(Y-T),3)/K;
%             meanAbsoluteError = sum(( abs(Y-T).*(T>0.0001) ),3)/K;
            
            % Take mean over mini-batch
            N = size(Y,4);
            loss = sum(sum(sum(meanAbsoluteError)))/N;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Returns the derivatives of the MAE loss with respect to the predictions Y

            N = size(Y,4);
            dLdY = sign(Y-T)/N;
%             e = ( sign(Y-T).*(T>0.0001) );
%             dLdY = Y-T;%e-(e.*(e>0).*0.9999);%/N;
            
        end
    end
end