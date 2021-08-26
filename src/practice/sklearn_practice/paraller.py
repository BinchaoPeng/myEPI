with parallel:
    all_candidate_params = []
    all_out = []
    all_more_results = defaultdict(list)


    def evaluate_candidates(candidate_params, cv=None,
                            more_results=None):
        cv = cv or cv_orig
        candidate_params = list(candidate_params)
        n_candidates = len(candidate_params)

        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates,"
                  " totalling {2} fits".format(
                n_splits, n_candidates, n_candidates * n_splits))

        out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                               X, y,
                                               train=train, test=test,
                                               parameters=parameters,
                                               split_progress=(
                                                   split_idx,
                                                   n_splits),
                                               candidate_progress=(
                                                   cand_idx,
                                                   n_candidates),
                                               **fit_and_score_kwargs)
                       for (cand_idx, parameters),
                           (split_idx, (train, test)) in product(
            enumerate(candidate_params),
            enumerate(cv.split(X, y, groups))))

        if len(out) < 1:
            raise ValueError('No fits were performed. '
                             'Was the CV iterator empty? '
                             'Were there no candidates?')
        elif len(out) != n_candidates * n_splits:
            raise ValueError('cv.split and cv.get_n_splits returned '
                             'inconsistent results. Expected {} '
                             'splits, got {}'
                             .format(n_splits,
                                     len(out) // n_candidates))

        # For callable self.scoring, the return type is only know after
        # calling. If the return type is a dictionary, the error scores
        # can now be inserted with the correct key. The type checking
        # of out will be done in `_insert_error_scores`.
        if callable(self.scoring):
            _insert_error_scores(out, self.error_score)
        all_candidate_params.extend(candidate_params)
        all_out.extend(out)
        if more_results is not None:
            for key, value in more_results.items():
                all_more_results[key].extend(value)

        nonlocal results
        results = self._format_results(
            all_candidate_params, n_splits, all_out,
            all_more_results)

        return results


    self._run_search(evaluate_candidates)

    # multimetric is determined here because in the case of a callable
    # self.scoring the return type is only known after calling
    first_test_score = all_out[0]['test_scores']
    self.multimetric_ = isinstance(first_test_score, dict)

    # check refit_metric now for a callabe scorer that is multimetric
    if callable(self.scoring) and self.multimetric_:
        self._check_refit_for_multimetric(first_test_score)
        refit_metric = self.refit