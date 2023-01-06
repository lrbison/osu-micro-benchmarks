#define BENCHMARK "OSU MPI_Fetch_and_op%s latency Test"
/*
 * Copyright (C) 2003-2022 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <osu_util_mpi.h>
#include <errno.h>

double  t_start = 0.0, t_end = 0.0;
uint64_t *sbuf=NULL, *tbuf=NULL, *win_base = NULL;
omb_graph_options_t omb_graph_op;

void gdb_attach(const char* env_variable) {
    char hostname[256];
    char *val = getenv(env_variable);
    if (!val || val[0] != 'y') return;
    
    size_t attach_pid = getpid();
    gethostname(hostname, sizeof(hostname));
    // if (fork() == 0) {
    //     char pid_str[12];
    //     sprintf(pid_str, "%d",attach_pid);
    //     printf("on %s I will launch gdbserver and attach to %s\n", hostname, pid_str);
    //     int prob = execlp("/usr/bin/gdbserver","/usr/bin/gdbserver","dummy:2345","--attach",pid_str);
    //     printf("Problem was %d, errno=%d\n",prob,errno);
    //     perror(NULL);
    // }
    // else {
        volatile int i = 0;
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i)
            sleep(5); // gdb: set var i = 7
    // }
}

void print_latency (int, int);
void run_fop_with_lock (int, enum WINDOW);
void run_fop_with_fence (int, enum WINDOW);
void run_fop_with_lock_all (int, enum WINDOW);
void run_fop_with_flush (int, enum WINDOW);
void run_fop_with_flush_local (int, enum WINDOW);
void run_fop_with_pscw(int rank, enum WINDOW win_type, MPI_Datatype data_type, MPI_Op op);

int main (int argc, char *argv[])
{
    int         rank,nprocs;
    int         po_ret = PO_OKAY;

    options.win = WIN_ALLOCATE;
    options.sync = FLUSH;

    options.bench = ONE_SIDED;
    options.subtype = LAT;
    options.synctype = ALL_SYNC;

    MPI_Datatype dtype_list[14] = {MPI_CHAR, MPI_UNSIGNED_CHAR,  MPI_SHORT, MPI_UNSIGNED_SHORT,
                                  MPI_INT, MPI_UNSIGNED, MPI_LONG_LONG, MPI_UNSIGNED_LONG_LONG,
                                  MPI_FLOAT, MPI_DOUBLE, MPI_LONG_DOUBLE,
                                  MPI_C_FLOAT_COMPLEX, MPI_C_DOUBLE_COMPLEX, MPI_C_LONG_DOUBLE_COMPLEX};
    MPI_Op op_list[4] = {MPI_MAX, MPI_MIN, MPI_SUM, MPI_PROD};


    set_header(HEADER);
    set_benchmark_name("osu_fop_latency");

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    gdb_attach("DEBUG");

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (0 == rank) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(rank);
            case PO_HELP_MESSAGE:
                usage_one_sided("osu_fop_latency");
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(rank);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }

    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if (nprocs != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());

        return EXIT_FAILURE;
    }


    for (int jdata_type = 0; jdata_type < sizeof(dtype_list); jdata_type++) {
    for (int jop = 0; jop < sizeof(op_list); jop++) {

    print_header_one_sided(rank, options.win, options.sync);

    //if (jop > 0) break;
    if (jdata_type > 0) break;

    switch (options.sync) {
        case LOCK:
            run_fop_with_lock(rank, options.win);
            break;
        case LOCK_ALL:
            run_fop_with_lock_all(rank, options.win);
            break;
        case PSCW:
            run_fop_with_pscw(rank, options.win, dtype_list[jdata_type], op_list[jop]);
            break;
        case FENCE:
            run_fop_with_fence(rank, options.win);
            break;
        case FLUSH_LOCAL:
            run_fop_with_flush_local(rank, options.win);
            break;
        default:
            run_fop_with_flush(rank, options.win);
            break;
    }

    }}

    for (int jrank_print=0; jrank_print<2; jrank_print++) {
        if (jrank_print == rank) {
            printf("-------------------------------------------\n");
            printf("Atomic Data Validation results for Rank=%d:\n",rank);
            atomic_data_validation_print_summary();
            printf("-------------------------------------------\n");
        }
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
    


    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }
    return EXIT_SUCCESS;
}

void print_latency(int rank, int size)
{
    if (rank == 0) {
        fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                FLOAT_PRECISION, (t_end - t_start) * 1.0e6 / options.iterations);
        fflush(stdout);
    }
}

/*Run FOP with flush local*/
void run_fop_with_flush_local (int rank, enum WINDOW type)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i;
    int jrank;
    MPI_Win     win;

    MPI_Aint disp = 0;

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data,
            &omb_graph_op, 8, options.iterations);
    omb_papi_init(&papi_eventset);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    allocate_atomic_memory(rank, (char **)&sbuf,
            (char **)&tbuf, NULL, (char **)&win_base, options.max_message_size, type, &win);

    atomic_data_validation_setup(MPI_LONG_LONG, rank, win_base, options.max_message_size);
    atomic_data_validation_setup(MPI_LONG_LONG, rank, sbuf, options.max_message_size);
    atomic_data_validation_setup(MPI_LONG_LONG, rank, tbuf, options.max_message_size);


    if (rank == 0) {
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
        MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime ();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            printf("BEFORE: Tbuf is %lld\n",*(long long*)tbuf);
            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, MPI_LONG_LONG, 1, disp, MPI_SUM, win));
            printf("AFTER1: Tbuf is %lld\n",*(long long*)tbuf);
            MPI_CHECK(MPI_Win_flush_local(1, win));
            printf("AFTER2: Tbuf is %lld\n",*(long long*)tbuf);
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] = (t_graph_end -
                            t_graph_start) * 1.0e6;
                }

                atomic_data_validation_check(MPI_LONG_LONG, MPI_SUM, rank, win_base, sbuf, 32, 0, 1);
            }
        }
        t_end = MPI_Wtime ();
        MPI_CHECK(MPI_Win_unlock(1, win));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, 8);
    print_latency(rank, 8);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory (sbuf, win_base, tbuf, NULL, type, win, rank);

    atomic_data_validation_print_summary();
}

/*Run FOP with flush */
void run_fop_with_flush (int rank, enum WINDOW type)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data,
            &omb_graph_op, 8, options.iterations);
    omb_papi_init(&papi_eventset);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    allocate_atomic_memory(rank, (char **)&sbuf,
            (char **)&tbuf, NULL, (char **)&win_base, options.max_message_size, type, &win);

    if (rank == 0) {
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }
        MPI_CHECK(MPI_Win_lock(MPI_LOCK_SHARED, 1, 0, win));
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime ();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, MPI_LONG_LONG, 1, disp, MPI_SUM, win));
            MPI_CHECK(MPI_Win_flush(1, win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] = (t_graph_end -
                            t_graph_start) * 1.0e6;
                }
            }
        }
        t_end = MPI_Wtime ();
        MPI_CHECK(MPI_Win_unlock(1, win));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, 8);
    print_latency(rank, 8);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory (sbuf, win_base, tbuf, NULL, type, win, rank);
}

/*Run FOP with Lock_all/unlock_all */
void run_fop_with_lock_all (int rank, enum WINDOW type)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data,
            &omb_graph_op, 8, options.iterations);
    omb_papi_init(&papi_eventset);
    allocate_atomic_memory(rank, (char **)&sbuf,
            (char **)&tbuf, NULL, (char **)&win_base, options.max_message_size, type, &win);

    if (rank == 0) {
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime ();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Win_lock_all(0, win));
            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, MPI_LONG_LONG, 1, disp, MPI_SUM, win));
            MPI_CHECK(MPI_Win_unlock_all(win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] = (t_graph_end -
                            t_graph_start) * 1.0e6;
                }
            }
        }
        t_end = MPI_Wtime ();
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, 8);
    print_latency(rank, 8);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory (sbuf, win_base, tbuf, NULL, type, win, rank);
}

/*Run FOP with Lock/unlock */
void run_fop_with_lock(int rank, enum WINDOW type)
{
    int i;
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    MPI_Aint disp = 0;
    MPI_Win     win;

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data,
            &omb_graph_op, 8, options.iterations);
    omb_papi_init(&papi_eventset);
    allocate_atomic_memory(rank, (char **)&sbuf,
            (char **)&tbuf, NULL, (char **)&win_base, options.max_message_size, type, &win);

    if (rank == 0) {
        if (type == WIN_DYNAMIC) {
            disp = disp_remote;
        }

        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime ();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 1, 0, win));
            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, MPI_LONG_LONG, 1, disp, MPI_SUM, win));
            MPI_CHECK(MPI_Win_unlock(1, win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] = (t_graph_end -
                            t_graph_start) * 1.0e6;
                }
            }
        }
        t_end = MPI_Wtime ();
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, 8);
    print_latency(rank, 8);
    if (options.graph && 0 == rank) {
        omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations;
    }
    omb_graph_plot(&omb_graph_op, benchmark_name);
    omb_graph_free_data_buffers(&omb_graph_op);
    omb_papi_free(&papi_eventset);
    free_atomic_memory (sbuf, win_base, tbuf, NULL, type, win, rank);
}

/*Run FOP with Fence */
void run_fop_with_fence(int rank, enum WINDOW type)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    allocate_atomic_memory(rank, (char **)&sbuf,
            (char **)&tbuf, NULL, (char **)&win_base, options.max_message_size, type, &win);

    if (type == WIN_DYNAMIC) {
        disp = disp_remote;
    }
    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data,
            &omb_graph_op, 8, options.iterations);
    omb_papi_init(&papi_eventset);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == 0) {
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime ();
            }
            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, MPI_LONG_LONG, 1, disp, MPI_SUM, win));
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(MPI_Win_fence(0, win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] = (t_graph_end -
                            t_graph_start) * 1.0e6;
                }
            }
        }
        t_end = MPI_Wtime ();
    } else {
        for (i = 0; i < options.skip + options.iterations; i++) {
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
            }
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(MPI_Win_fence(0, win));
            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, MPI_LONG_LONG, 0, disp, MPI_SUM, win));
            MPI_CHECK(MPI_Win_fence(0, win));
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, 8);
    if (rank == 0) {
        fprintf(stdout, "%-*d%*.*f\n", 10, 8, FIELD_WIDTH,
                FLOAT_PRECISION, (t_end - t_start) * 1.0e6 / options.iterations / 2);
        fflush(stdout);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg = (t_end - t_start) * 1.0e6 /
                options.iterations/ 2;
        }
        omb_graph_plot(&omb_graph_op, benchmark_name);
        omb_graph_free_data_buffers(&omb_graph_op);
    }

    omb_papi_free(&papi_eventset);
    free_atomic_memory (sbuf, win_base, tbuf, NULL, type, win, rank);
    atomic_data_validation_print_summary();
}

/*Run FOP with Post/Start/Complete/Wait */
void run_fop_with_pscw(int rank, enum WINDOW win_type, MPI_Datatype data_type, MPI_Op op)
{
    double t_graph_start = 0.0, t_graph_end = 0.0;
    omb_graph_data_t *omb_graph_data = NULL;
    int papi_eventset = OMB_PAPI_NULL;
    int destrank, i;
    MPI_Aint disp = 0;
    MPI_Win     win;

    MPI_Group       comm_group, group;
    MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &comm_group));

    omb_graph_op.number_of_graphs = 0;
    omb_graph_allocate_and_get_data_buffer(&omb_graph_data,
            &omb_graph_op, 8, options.iterations);
    omb_papi_init(&papi_eventset);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    allocate_atomic_memory(rank, (char **)&sbuf,
        (char **)&tbuf, NULL, (char **)&win_base, options.max_message_size, win_type, &win);


    if (win_type == WIN_DYNAMIC) {
        disp = disp_remote;
    }

    if (rank == 0) {
        destrank = 1;
        MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        for (i = 0; i < options.skip + options.iterations; i++) {
            printf("Rank=%d, Iteration %d\n",rank,i);
            atomic_data_validation_setup(data_type, rank, sbuf, options.max_message_size);
            atomic_data_validation_setup(data_type, rank, tbuf, options.max_message_size);
            atomic_data_validation_setup(data_type, rank, win_base, options.max_message_size);
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            MPI_CHECK(MPI_Win_start (group, 0, win));

            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
                t_start = MPI_Wtime ();
            }

            if (i >= options.skip) {
                t_graph_start = MPI_Wtime();
            }
            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, data_type, 1, disp, op, win));
            MPI_CHECK(MPI_Win_complete(win));
            MPI_CHECK(MPI_Win_post(group, 0, win));
            MPI_CHECK(MPI_Win_wait(win));
            if (i >= options.skip) {
                t_graph_end = MPI_Wtime();
                if (options.graph) {
                    omb_graph_data->data[i - options.skip] = (t_graph_end -
                            t_graph_start) * 1.0e6;
                }
            }
            atomic_data_validation_check(data_type, op, rank, win_base, tbuf, options.max_message_size, 1, 1);
        }

        t_end = MPI_Wtime ();
    } else {
        /* rank=1 */
        destrank = 0;

        MPI_CHECK(MPI_Group_incl(comm_group, 1, &destrank, &group));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        for (i = 0; i < options.skip + options.iterations; i++) {
            printf("Rank=%d, Iteration %d\n",rank,i);
            atomic_data_validation_setup(data_type, rank, sbuf, options.max_message_size);
            atomic_data_validation_setup(data_type, rank, tbuf, options.max_message_size);
            atomic_data_validation_setup(data_type, rank, win_base, options.max_message_size);
            MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
            if (i == options.skip) {
                omb_papi_start(&papi_eventset);
            }
            MPI_CHECK(MPI_Win_post(group, 0, win));
            MPI_CHECK(MPI_Win_wait(win));
            MPI_CHECK(MPI_Win_start(group, 0, win));

            MPI_CHECK(MPI_Fetch_and_op(sbuf, tbuf, data_type, 0, disp, op, win));
            MPI_CHECK(MPI_Win_complete(win));

            atomic_data_validation_check(data_type, op, rank, win_base, tbuf, options.max_message_size, 1, 1);
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    omb_papi_stop_and_print(&papi_eventset, 8);
    if (rank == 0) {
        fprintf(stdout, "%-*d%*.*f\n", 10, 8, FIELD_WIDTH,
                FLOAT_PRECISION, (t_end - t_start) * 1.0e6 / options.iterations / 2);
        fflush(stdout);
        if (options.graph && 0 == rank) {
            omb_graph_data->avg = (t_end - t_start) * 1.0e6 / options.iterations
                / 2;
        }
        omb_graph_plot(&omb_graph_op, benchmark_name);
        omb_graph_free_data_buffers(&omb_graph_op);
    }

    omb_papi_free(&papi_eventset);
    MPI_CHECK(MPI_Group_free(&group));
    MPI_CHECK(MPI_Group_free(&comm_group));

    free_atomic_memory (sbuf, win_base, tbuf, NULL, win_type, win, rank);

}
/* vi: set sw=4 sts=4 tw=80: */
