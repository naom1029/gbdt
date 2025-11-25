classdef Node < handle
   properties
      left
      right
      feature
      features
      threshold
      X
      depth
      is_leaf
      score
      
      
   end
   methods
       function obj = Node(X,depth)
           obj.left = [];
           obj.right = [];
           obj.feature = [];
           obj.threshold = [];
           obj.X = X;
           num_columns = size(X,2);
           obj.features = 1:num_columns;
           obj.depth = depth;
       end
   end
end