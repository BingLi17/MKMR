function problemtype = problem_type(loss);

    switch loss.type,
        case 'regression'
           problemtype='regression';
        case { 'logistic' , 'sparselogistic' }
           problemtype='classification';
        case { 'logistic_unbalanced' }
           problemtype='classification_unbalanced';
    end
