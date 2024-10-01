classdef Node < handle
   properties
      left
      right
      feature
      features
      threshold
      X
      Y
      depth
      class_table
      class_label
      is_leaf
      score
      
      
   end
   methods
       function obj = Node(X,Y,depth)
           obj.left = [];
           obj.right = [];
           obj.feature = [];
           obj.threshold = [];
           obj.X = X;
           obj.Y = Y;
           num_columns = size(X,2);
           obj.features = 1:num_columns;
           obj.depth = depth;
           obj.class_table =  [sum(Y == 0), sum(Y == 1), sum(Y == 2)];
           [~,obj.class_label] = max(obj.class_table);
           
       end
   end
end