  // Quick gemm if alpha==0.0
  if ( alpha==zero ) {
    if ( beta==zero )
      if ( orderC=='C' ) {
        #pragma omp parallel
        for (int jc=0; jc<n; jc++ )
          for (int ic=0; ic<m; ic++ )
            Ccol(ic,jc) = 0.0;
      }
      else {
        #pragma omp parallel
        for (int jc=0; jc<n; jc++ )
          for (int ic=0; ic<m; ic++ )
            Crow(ic,jc) = 0.0;
      }
    else
      if ( orderC=='C' ) {
        #pragma omp parallel
        for (int jc=0; jc<n; jc++ )
          for (int ic=0; ic<m; ic++ )
            Ccol(ic,jc) = beta*Ccol(ic,jc);
      }
      else {
        #pragma omp parallel
        for (int jc=0; jc<n; jc++ )
          for (int ic=0; ic<m; ic++ )
            Crow(ic,jc) = beta*Crow(ic,jc);
      }
    return;
  }
