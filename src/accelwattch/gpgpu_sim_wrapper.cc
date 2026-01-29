// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, Ahmed ElTantawy,
// Vijay Kandiah, Nikos Hardavellas The University of British Columbia,
// Northwestern University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpgpu_sim_wrapper.h"
#include <sys/stat.h>
#define SP_BASE_POWER 0
#define SFU_BASE_POWER 0

static const char* pwr_cmp_label[] = {
    "IBP,",        "ICP,",        "DCP,",      "TCP,",      "CCP,",
    "SHRDP,",      "RFP,",        "INTP,",     "FPUP,",     "DPUP,",
    "INT_MUL24P,", "INT_MUL32P,", "INT_MULP,", "INT_DIVP,", "FP_MULP,",
    "FP_DIVP,",    "FP_SQRTP,",   "FP_LGP,",   "FP_SINP,",  "FP_EXP,",
    "DP_MULP,",    "DP_DIVP,",    "TENSORP,",  "TEXP,",     "SCHEDP,",
    "L2CP,",       "MCP,",        "NOCP,",     "DRAMP,",    "PIPEP,",
    "IDLE_COREP,", "CONSTP",      "STATICP"};

enum pwr_cmp_t {
  IBP = 0,
  ICP,
  DCP,
  TCP,
  CCP,
  SHRDP,
  RFP,
  INTP,
  FPUP,
  DPUP,
  INT_MUL24P,
  INT_MUL32P,
  INT_MULP,
  INT_DIVP,
  FP_MULP,
  FP_DIVP,
  FP_SQRTP,
  FP_LGP,
  FP_SINP,
  FP_EXP,
  DP_MULP,
  DP_DIVP,
  TENSORP,
  TEXP,
  SCHEDP,
  L2CP,
  MCP,
  NOCP,
  DRAMP,
  PIPEP,
  IDLE_COREP,
  CONSTP,
  STATICP,
  NUM_COMPONENTS_MODELLED
};

gpgpu_sim_wrapper::gpgpu_sim_wrapper(bool power_simulation_enabled,
                                     char* xmlfile, int power_simulation_mode,
                                     bool dvfs_enabled, unsigned n_shader) {
  kernel_sample_count = 0;
  total_sample_count = 0;

  kernel_tot_power = 0;
  avg_threads_per_warp_tot = 0;
  num_pwr_cmps = NUM_COMPONENTS_MODELLED;
  num_perf_counters = NUM_PERFORMANCE_COUNTERS;

  // Initialize per-component counter/power vectors
  avg_max_min_counters<double> init;
  kernel_cmp_pwr.resize(NUM_COMPONENTS_MODELLED, init);
  kernel_cmp_perf_counters.resize(NUM_PERFORMANCE_COUNTERS, init);

  kernel_power = init;   // Per-kernel powers
  gpu_tot_power = init;  // Global powers

  sample_cmp_pwr.resize(NUM_COMPONENTS_MODELLED, 0);
  SM_core_power.resize(n_shader, 0);

  sample_perf_counters.resize(NUM_PERFORMANCE_COUNTERS, 0);
  sample_Per_cmp_pwr.resize(n_shader);
  sample_Per_perf_counters.resize(n_shader);
  for (unsigned i = 0; i < n_shader; i++) {
    sample_Per_cmp_pwr[i].resize(NUM_COMPONENTS_MODELLED, 0);
    sample_Per_perf_counters[i].resize(NUM_PERFORMANCE_COUNTERS, 0);
  }
  initpower_coeff.resize(NUM_PERFORMANCE_COUNTERS, 0);
  effpower_coeff.resize(NUM_PERFORMANCE_COUNTERS, 0);
  #if Lyhong_Percore_sim
  initpower_coeff_per_core.assign(n_shader, std::vector<double>(NUM_PERFORMANCE_COUNTERS, 0.0));
  effpower_coeff_per_core.assign(n_shader, std::vector<double>(NUM_PERFORMANCE_COUNTERS, 0.0));
  #endif
  const_dynamic_power = 0;
  proc_power = 0;

  sm_header_dumped = false;
  g_power_filename = NULL;
  g_power_trace_filename = NULL;
  lyhong_filename_interface = NULL;
  g_metric_trace_filename = NULL;
  g_steady_state_tracking_filename = NULL;
  xml_filename = xmlfile;
  g_power_simulation_enabled = power_simulation_enabled;
  g_power_simulation_mode = power_simulation_mode;
  g_dvfs_enabled = dvfs_enabled;
  g_power_trace_enabled = false;
  g_steady_power_levels_enabled = false;
  g_power_trace_zlevel = 0;
  g_power_per_cycle_dump = false;
  gpu_steady_power_deviation = 0;
  gpu_steady_min_period = 0;

  gpu_stat_sample_freq = 0;
  p = new ParseXML();
  if (g_power_simulation_enabled) {
    p->parse(xml_filename);
  }
  proc = new Processor(p);
  power_trace_file = NULL;
  metric_trace_file = NULL;
  steady_state_tacking_file = NULL;
  has_written_avg = false;
  init_inst_val = false;
}

gpgpu_sim_wrapper::~gpgpu_sim_wrapper() {}

bool gpgpu_sim_wrapper::sanity_check(double a, double b) {
  if (b == 0)
    return (abs(a - b) < 0.00001);
  else
    return (abs(a - b) / abs(b) < 0.00001);

  return false;
}
void gpgpu_sim_wrapper::init_mcpat_hw_mode(unsigned gpu_sim_cycle) {
  p->sys.total_cycles =
      gpu_sim_cycle;  // total simulated cycles for current kernel
}

void gpgpu_sim_wrapper::init_mcpat(
    char* xmlfile, char* powerfilename, char* power_trace_filename,
    char* metric_trace_filename, char* steady_state_filename,
    bool power_sim_enabled, bool trace_enabled, bool steady_state_enabled,
    bool power_per_cycle_dump, double steady_power_deviation,
    double steady_min_period, int zlevel, double init_val, int stat_sample_freq,
    int power_sim_mode, bool dvfs_enabled, unsigned clock_freq,
    unsigned num_shaders, char* lyhong_filename, char* lyhong_SM_filename) {
  // Write File Headers for (-metrics trace, -power trace)

  reset_counters();
  static bool mcpat_init = true;

  // initialize file name if it is not set
  time_t curr_time;
  time(&curr_time);
  char* date = ctime(&curr_time);
  char* s = date;
  while (*s) {
    if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
    if (*s == '\n' || *s == '\r') *s = 0;
    s++;
  }

  if (mcpat_init) {
    lyhong_filename_interface = lyhong_filename;
    lyhong_SM_filename_interface = lyhong_SM_filename;
    g_power_filename = powerfilename;
    g_power_trace_filename = power_trace_filename;
    g_metric_trace_filename = metric_trace_filename;
    g_steady_state_tracking_filename = steady_state_filename;
    xml_filename = xmlfile;
    g_power_simulation_enabled = power_sim_enabled;
    g_power_simulation_mode = power_sim_mode;
    g_dvfs_enabled = dvfs_enabled;
    g_power_trace_enabled = trace_enabled;
    g_steady_power_levels_enabled = steady_state_enabled;
    g_power_trace_zlevel = zlevel;
    g_power_per_cycle_dump = power_per_cycle_dump;
    gpu_steady_power_deviation = steady_power_deviation;
    gpu_steady_min_period = steady_min_period;

    gpu_stat_sample_freq = stat_sample_freq;

    // p->sys.total_cycles=gpu_stat_sample_freq*4;
    p->sys.total_cycles = gpu_stat_sample_freq;
    p->sys.target_core_clockrate = clock_freq;
    p->sys.number_of_cores = num_shaders;
    p->sys.core[0].clock_rate = clock_freq;
    power_trace_file = NULL;
    metric_trace_file = NULL;
    steady_state_tacking_file = NULL;

    if (g_power_trace_enabled) {
      power_trace_file = gzopen(g_power_trace_filename, "w");
      metric_trace_file = gzopen(g_metric_trace_filename, "w");
      if ((power_trace_file == NULL) || (metric_trace_file == NULL)) {
        printf("error - could not open trace files \n");
        exit(1);
      }
      gzsetparams(power_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);

      gzprintf(power_trace_file, "power,");
      for (unsigned i = 0; i < num_pwr_cmps; i++) {
        gzprintf(power_trace_file, pwr_cmp_label[i]);
      }
      gzprintf(power_trace_file, "\n");

      gzsetparams(metric_trace_file, g_power_trace_zlevel, Z_DEFAULT_STRATEGY);
      for (unsigned i = 0; i < num_perf_counters; i++) {
        gzprintf(metric_trace_file, perf_count_label[i]);
      }
      gzprintf(metric_trace_file, "\n");

      gzclose(power_trace_file);
      gzclose(metric_trace_file);
    }
    if (g_steady_power_levels_enabled) {
      steady_state_tacking_file = gzopen(g_steady_state_tracking_filename, "w");
      if ((steady_state_tacking_file == NULL)) {
        printf("error - could not open trace files \n");
        exit(1);
      }
      gzsetparams(steady_state_tacking_file, g_power_trace_zlevel,
                  Z_DEFAULT_STRATEGY);
      gzprintf(steady_state_tacking_file, "start,end,power,IPC,");
      for (unsigned i = 0; i < num_perf_counters; i++) {
        gzprintf(steady_state_tacking_file, perf_count_label[i]);
      }
      gzprintf(steady_state_tacking_file, "\n");

      gzclose(steady_state_tacking_file);
    }

    mcpat_init = false;
    has_written_avg = false;
    powerfile.open(g_power_filename);
    int flg = chmod(g_power_filename, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    assert(flg == 0);
  }
  lyhong_file.open(lyhong_filename_interface);
  if(!lyhong_file.is_open()) {
    powerfile << "lyhong_print: Error - could not open lyhong_file. " << std::endl; 
  }
  int lyhong_flg = chmod(lyhong_filename_interface, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  assert(lyhong_flg == 0);
  if(Lyhong_Percore_sim) {
    lyhong_SM_file.open(lyhong_SM_filename_interface);
    if(!lyhong_SM_file.is_open()) {
      powerfile << "lyhong_SM_print: Error - could not open lyhong_SM_file. " << std::endl; 
    }
    int lyhong_SM_flg = chmod(lyhong_SM_filename_interface, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    assert(lyhong_SM_flg == 0);
  }
  sample_val = 0;
  init_inst_val = init_val;  // gpu_tot_sim_insn+gpu_sim_insn;
}

void gpgpu_sim_wrapper::reset_counters() {
  avg_max_min_counters<double> init;
  for (unsigned i = 0; i < num_perf_counters; ++i) {
    sample_perf_counters[i] = 0;
    kernel_cmp_perf_counters[i] = init;
  }
  for (unsigned i = 0; i < num_pwr_cmps; ++i) {
    sample_cmp_pwr[i] = 0;
    kernel_cmp_pwr[i] = init;
  }

  // Reset per-kernel counters
  kernel_sample_count = 0;
  kernel_tot_power = 0;
  kernel_power = init;
  avg_threads_per_warp_tot = 0;
  return;
}

void gpgpu_sim_wrapper::set_inst_power(bool clk_gated_lanes, double tot_cycles,
                                       double busy_cycles, double tot_inst,
                                       double int_inst, double fp_inst,
                                       double load_inst, double store_inst,
                                       double committed_inst) {
  p->sys.core[0].gpgpu_clock_gated_lanes = clk_gated_lanes;
  p->sys.core[0].total_cycles = tot_cycles;
  p->sys.core[0].busy_cycles = busy_cycles;
  p->sys.core[0].total_instructions =
      tot_inst * p->sys.scaling_coefficients[TOT_INST];
  p->sys.core[0].int_instructions =
      int_inst * p->sys.scaling_coefficients[FP_INT];
  p->sys.core[0].fp_instructions =
      fp_inst * p->sys.scaling_coefficients[FP_INT];
  p->sys.core[0].load_instructions = load_inst;
  p->sys.core[0].store_instructions = store_inst;
  p->sys.core[0].committed_instructions = committed_inst;
  sample_perf_counters[FP_INT] = int_inst + fp_inst;
  sample_perf_counters[TOT_INST] = tot_inst;
}

// Lyhong_TODO: L1 cache is not modeled per core yet
void gpgpu_sim_wrapper::set_Per_inst_power(bool clk_gated_lanes, double tot_cycles,
                                          double busy_cycles, const std::vector<double> &Per_tot_inst,
                                          const std::vector<double> &Per_int_inst, const std::vector<double> &Per_fp_inst,
                                          double Per_load_inst, double Per_store_inst,
                                          const std::vector<double> &Per_committed_inst) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].gpgpu_clock_gated_lanes = clk_gated_lanes;
    p->sys.core[i].total_cycles = tot_cycles;
    p->sys.core[i].busy_cycles  = busy_cycles;
    p->sys.core[i].total_instructions =
        Per_tot_inst[i] * p->sys.scaling_coefficients[TOT_INST];
    p->sys.core[i].int_instructions =
        Per_int_inst[i] * p->sys.scaling_coefficients[FP_INT];
    p->sys.core[i].fp_instructions =
        Per_fp_inst[i] * p->sys.scaling_coefficients[FP_INT];
    p->sys.core[i].committed_instructions = Per_committed_inst[i];
    sample_Per_perf_counters[i][FP_INT] = Per_int_inst[i] + Per_fp_inst[i];
    sample_Per_perf_counters[i][TOT_INST] = Per_tot_inst[i];

    // L1 cache is not distributed to each core or precisely allocated to each core. Is precision required?
    p->sys.core[i].load_instructions  = Per_load_inst;
    p->sys.core[i].store_instructions = Per_store_inst;
  }
}

void gpgpu_sim_wrapper::set_regfile_power(double reads, double writes,
                                          double ops) {
  p->sys.core[0].int_regfile_reads =
      reads * p->sys.scaling_coefficients[REG_RD];
  p->sys.core[0].int_regfile_writes =
      writes * p->sys.scaling_coefficients[REG_WR];
  p->sys.core[0].non_rf_operands =
      ops * p->sys.scaling_coefficients[NON_REG_OPs];
  sample_perf_counters[REG_RD] = reads;
  sample_perf_counters[REG_WR] = writes;
  sample_perf_counters[NON_REG_OPs] = ops;
}
void gpgpu_sim_wrapper::set_Per_regfile_power(const std::vector<double> &Per_reads,
                                              const std::vector<double> &Per_writes,
                                              const std::vector<double> &Per_ops) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].int_regfile_reads =
        Per_reads[i] * p->sys.scaling_coefficients[REG_RD];
    p->sys.core[i].int_regfile_writes = 
        Per_writes[i] * p->sys.scaling_coefficients[REG_WR];
    p->sys.core[i].non_rf_operands =
        Per_ops[i] * p->sys.scaling_coefficients[NON_REG_OPs];
    sample_Per_perf_counters[i][REG_RD] = Per_reads[i];
    sample_Per_perf_counters[i][REG_WR] = Per_writes[i];
    sample_Per_perf_counters[i][NON_REG_OPs] = Per_ops[i];
  }
}

void gpgpu_sim_wrapper::set_icache_power(double hits, double misses) {
  p->sys.core[0].icache.read_accesses =
      hits * p->sys.scaling_coefficients[IC_H] +
      misses * p->sys.scaling_coefficients[IC_M];
  p->sys.core[0].icache.read_misses =
      misses * p->sys.scaling_coefficients[IC_M];
  sample_perf_counters[IC_H] = hits;
  sample_perf_counters[IC_M] = misses;
}
void gpgpu_sim_wrapper::set_Per_icache_power(double Per_hits,
                                            double Per_misses) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].icache.read_accesses =
        Per_hits * p->sys.scaling_coefficients[IC_H] +
        Per_misses * p->sys.scaling_coefficients[IC_M];
    p->sys.core[i].icache.read_misses =
        Per_misses * p->sys.scaling_coefficients[IC_M];
    sample_Per_perf_counters[i][IC_H] = Per_hits;
    sample_Per_perf_counters[i][IC_M] = Per_misses;
  }
}

void gpgpu_sim_wrapper::set_ccache_power(double hits, double misses) {
  p->sys.core[0].ccache.read_accesses =
      hits * p->sys.scaling_coefficients[CC_H] +
      misses * p->sys.scaling_coefficients[CC_M];
  p->sys.core[0].ccache.read_misses =
      misses * p->sys.scaling_coefficients[CC_M];
  sample_perf_counters[CC_H] = hits;
  sample_perf_counters[CC_M] = misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
}
void gpgpu_sim_wrapper::set_Per_ccache_power(const std::vector<double> &Per_hits,
                                             const std::vector<double> &Per_misses) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].ccache.read_accesses =
        Per_hits[i] * p->sys.scaling_coefficients[CC_H] +
        Per_misses[i] * p->sys.scaling_coefficients[CC_M];
    p->sys.core[i].ccache.read_misses =
        Per_misses[i] * p->sys.scaling_coefficients[CC_M];
    sample_Per_perf_counters[i][CC_H] = Per_hits[i];
    sample_Per_perf_counters[i][CC_M] = Per_misses[i];
  }
}

void gpgpu_sim_wrapper::set_tcache_power(double hits, double misses) {
  p->sys.core[0].tcache.read_accesses =
      hits * p->sys.scaling_coefficients[TC_H] +
      misses * p->sys.scaling_coefficients[TC_M];
  p->sys.core[0].tcache.read_misses =
      misses * p->sys.scaling_coefficients[TC_M];
  sample_perf_counters[TC_H] = hits;
  sample_perf_counters[TC_M] = misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
}
void gpgpu_sim_wrapper::set_Per_tcache_power(double Per_hits,
                                            double Per_misses) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].tcache.read_accesses =
        Per_hits * p->sys.scaling_coefficients[TC_H] +
        Per_misses * p->sys.scaling_coefficients[TC_M];
    p->sys.core[i].tcache.read_misses =
        Per_misses * p->sys.scaling_coefficients[TC_M];
    sample_Per_perf_counters[i][TC_H] = Per_hits;
    sample_Per_perf_counters[i][TC_M] = Per_misses;
  }
}

void gpgpu_sim_wrapper::set_shrd_mem_power(double accesses) {
  p->sys.core[0].sharedmemory.read_accesses =
      accesses * p->sys.scaling_coefficients[SHRD_ACC];
  sample_perf_counters[SHRD_ACC] = accesses;
}
void gpgpu_sim_wrapper::set_Per_shrd_mem_power(const std::vector<double> &Per_accesses) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].sharedmemory.read_accesses =
        Per_accesses[i] * p->sys.scaling_coefficients[SHRD_ACC];
    sample_Per_perf_counters[i][SHRD_ACC] = Per_accesses[i];
  }
}

void gpgpu_sim_wrapper::set_l1cache_power(double read_hits, double read_misses,
                                          double write_hits,
                                          double write_misses) {
  p->sys.core[0].dcache.read_accesses =
      read_hits * p->sys.scaling_coefficients[DC_RH] +
      read_misses * p->sys.scaling_coefficients[DC_RM];
  p->sys.core[0].dcache.read_misses =
      read_misses * p->sys.scaling_coefficients[DC_RM];
  p->sys.core[0].dcache.write_accesses =
      write_hits * p->sys.scaling_coefficients[DC_WH] +
      write_misses * p->sys.scaling_coefficients[DC_WM];
  p->sys.core[0].dcache.write_misses =
      write_misses * p->sys.scaling_coefficients[DC_WM];
  sample_perf_counters[DC_RH] = read_hits;
  sample_perf_counters[DC_RM] = read_misses;
  sample_perf_counters[DC_WH] = write_hits;
  sample_perf_counters[DC_WM] = write_misses;
  // TODO: coalescing logic is counted as part of the caches power (this is not
  // valid for no-caches architectures)
}
// Lyhong_TODO: L1 cache is not distributed to each core or precisely allocated to each core. Is precision required?
void gpgpu_sim_wrapper::set_Per_l1cache_power(
    double Per_read_hits,
    double Per_read_misses,
    double Per_write_hits,
    double Per_write_misses) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].dcache.read_accesses =
        Per_read_hits * p->sys.scaling_coefficients[DC_RH] +
        Per_read_misses * p->sys.scaling_coefficients[DC_RM];
    p->sys.core[i].dcache.read_misses =
        Per_read_misses * p->sys.scaling_coefficients[DC_RM];
    p->sys.core[i].dcache.write_accesses =
        Per_write_hits * p->sys.scaling_coefficients[DC_WH] +
        Per_write_misses * p->sys.scaling_coefficients[DC_WM];
    p->sys.core[i].dcache.write_misses =
        Per_write_misses * p->sys.scaling_coefficients[DC_WM];
    sample_Per_perf_counters[i][DC_RH] = Per_read_hits;
    sample_Per_perf_counters[i][DC_RM] = Per_read_misses;
    sample_Per_perf_counters[i][DC_WH] = Per_write_hits;
    sample_Per_perf_counters[i][DC_WM] = Per_write_misses;
  }
}

void gpgpu_sim_wrapper::set_l2cache_power(double read_hits, double read_misses,
                                          double write_hits,
                                          double write_misses) {
  p->sys.l2.total_accesses = read_hits * p->sys.scaling_coefficients[L2_RH] +
                             read_misses * p->sys.scaling_coefficients[L2_RM] +
                             write_hits * p->sys.scaling_coefficients[L2_WH] +
                             write_misses * p->sys.scaling_coefficients[L2_WM];
  p->sys.l2.read_accesses = read_hits * p->sys.scaling_coefficients[L2_RH] +
                            read_misses * p->sys.scaling_coefficients[L2_RM];
  p->sys.l2.write_accesses = write_hits * p->sys.scaling_coefficients[L2_WH] +
                             write_misses * p->sys.scaling_coefficients[L2_WM];
  p->sys.l2.read_hits = read_hits * p->sys.scaling_coefficients[L2_RH];
  p->sys.l2.read_misses = read_misses * p->sys.scaling_coefficients[L2_RM];
  p->sys.l2.write_hits = write_hits * p->sys.scaling_coefficients[L2_WH];
  p->sys.l2.write_misses = write_misses * p->sys.scaling_coefficients[L2_WM];
  sample_perf_counters[L2_RH] = read_hits;
  sample_perf_counters[L2_RM] = read_misses;
  sample_perf_counters[L2_WH] = write_hits;
  sample_perf_counters[L2_WM] = write_misses;
}

void gpgpu_sim_wrapper::set_num_cores(double num_core) { 
  num_cores = num_core; 
  tot_fpu_accesses_PerCore.resize(num_cores);
  tot_sfu_accesses_PerCore.resize(num_cores);
}

void gpgpu_sim_wrapper::set_idle_core_power(double num_idle_core) {
  p->sys.num_idle_cores = num_idle_core;
  sample_perf_counters[IDLE_CORE_N] = num_idle_core;
  num_idle_cores = num_idle_core;
}
void gpgpu_sim_wrapper::set_Per_idle_core_power(const std::vector<float> &Per_active_core) {
  int temp_idle = 0;
  const float eps = 1e-6f;
  for(unsigned i = 0; i < num_cores; i++) {
    temp_idle = (Per_active_core[i] <= eps) ? 1 : 0;
    sample_Per_cmp_pwr[i][IDLE_COREP] = temp_idle * p->sys.idle_core_power;
  }
}

void gpgpu_sim_wrapper::set_duty_cycle_power(double duty_cycle) {
  p->sys.core[0].pipeline_duty_cycle =
      duty_cycle * p->sys.scaling_coefficients[PIPE_A];
  sample_perf_counters[PIPE_A] = duty_cycle;
}
void gpgpu_sim_wrapper::set_Per_duty_cycle_power(const std::vector<double> &Per_duty_cycle) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].pipeline_duty_cycle =
        Per_duty_cycle[i] * p->sys.scaling_coefficients[PIPE_A];
    sample_Per_perf_counters[i][PIPE_A] = Per_duty_cycle[i];
  }
}

void gpgpu_sim_wrapper::set_mem_ctrl_power(double reads, double writes,
                                           double dram_precharge) {
  p->sys.mc.memory_accesses = reads * p->sys.scaling_coefficients[MEM_RD] +
                              writes * p->sys.scaling_coefficients[MEM_WR];
  p->sys.mc.memory_reads = reads * p->sys.scaling_coefficients[MEM_RD];
  p->sys.mc.memory_writes = writes * p->sys.scaling_coefficients[MEM_WR];
  p->sys.mc.dram_pre = dram_precharge * p->sys.scaling_coefficients[MEM_PRE];
  sample_perf_counters[MEM_RD] = reads;
  sample_perf_counters[MEM_WR] = writes;
  sample_perf_counters[MEM_PRE] = dram_precharge;
}

void gpgpu_sim_wrapper::set_model_voltage(double model_voltage) {
  modeled_chip_voltage = model_voltage;
}

void gpgpu_sim_wrapper::set_exec_unit_power(double fpu_accesses,
                                            double ialu_accesses,
                                            double sfu_accesses) {
  p->sys.core[0].fpu_accesses = fpu_accesses;
  tot_fpu_accesses = fpu_accesses;
  // Integer ALU (not present in Tesla)
  p->sys.core[0].ialu_accesses = ialu_accesses;

  // Sfu accesses
  p->sys.core[0].mul_accesses = sfu_accesses;
  tot_sfu_accesses = sfu_accesses;
}
// Lyhong_TODO: tot_fpu_accesses and tot_sfu_accesses are not defined for per-core accesses. Is it required?
void gpgpu_sim_wrapper::set_Per_exec_unit_power(const std::vector<double> &Per_fpu_accesses,
                                               const std::vector<double> &Per_ialu_accesses,
                                               const std::vector<double> &Per_sfu_accesses) {
  tot_fpu_accesses = 0;
  tot_sfu_accesses = 0;
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].fpu_accesses = Per_fpu_accesses[i];
    p->sys.core[i].ialu_accesses = Per_ialu_accesses[i];
    p->sys.core[i].mul_accesses = Per_sfu_accesses[i];
    tot_fpu_accesses_PerCore[i] = Per_fpu_accesses[i];
    tot_sfu_accesses_PerCore[i] = Per_sfu_accesses[i];
    tot_fpu_accesses += Per_fpu_accesses[i];
    tot_sfu_accesses += Per_sfu_accesses[i];
  }
}

PowerscalingCoefficients* gpgpu_sim_wrapper::get_scaling_coeffs() {
  PowerscalingCoefficients* scalingCoeffs = new PowerscalingCoefficients();

  scalingCoeffs->int_coeff = p->sys.scaling_coefficients[INT_ACC];
  scalingCoeffs->int_mul_coeff = p->sys.scaling_coefficients[INT_MUL_ACC];
  scalingCoeffs->int_mul24_coeff = p->sys.scaling_coefficients[INT_MUL24_ACC];
  scalingCoeffs->int_mul32_coeff = p->sys.scaling_coefficients[INT_MUL32_ACC];
  scalingCoeffs->int_div_coeff = p->sys.scaling_coefficients[INT_DIV_ACC];
  scalingCoeffs->fp_coeff = p->sys.scaling_coefficients[FP_ACC];
  scalingCoeffs->dp_coeff = p->sys.scaling_coefficients[DP_ACC];
  scalingCoeffs->fp_mul_coeff = p->sys.scaling_coefficients[FP_MUL_ACC];
  scalingCoeffs->fp_div_coeff = p->sys.scaling_coefficients[FP_DIV_ACC];
  scalingCoeffs->dp_mul_coeff = p->sys.scaling_coefficients[DP_MUL_ACC];
  scalingCoeffs->dp_div_coeff = p->sys.scaling_coefficients[DP_DIV_ACC];
  scalingCoeffs->sqrt_coeff = p->sys.scaling_coefficients[FP_SQRT_ACC];
  scalingCoeffs->log_coeff = p->sys.scaling_coefficients[FP_LG_ACC];
  scalingCoeffs->sin_coeff = p->sys.scaling_coefficients[FP_SIN_ACC];
  scalingCoeffs->exp_coeff = p->sys.scaling_coefficients[FP_EXP_ACC];
  scalingCoeffs->tensor_coeff = p->sys.scaling_coefficients[TENSOR_ACC];
  scalingCoeffs->tex_coeff = p->sys.scaling_coefficients[TEX_ACC];
  return scalingCoeffs;
}

void gpgpu_sim_wrapper::set_int_accesses(double ialu_accesses,
                                         double imul24_accesses,
                                         double imul32_accesses,
                                         double imul_accesses,
                                         double idiv_accesses) {
  sample_perf_counters[INT_ACC] = ialu_accesses;
  sample_perf_counters[INT_MUL24_ACC] = imul24_accesses;
  sample_perf_counters[INT_MUL32_ACC] = imul32_accesses;
  sample_perf_counters[INT_MUL_ACC] = imul_accesses;
  sample_perf_counters[INT_DIV_ACC] = idiv_accesses;
}

void gpgpu_sim_wrapper::set_Per_int_accesses(const std::vector<double> &Per_imul24_accesses,
                                             const std::vector<double> &Per_imul32_accesses,
                                             const std::vector<double> &Per_idiv_accesses,
                                             const std::vector<double> &Per_imul_accesses) {
  for (unsigned i = 0; i < num_cores; i++) {
    sample_Per_perf_counters[i][INT_MUL24_ACC] = Per_imul24_accesses[i];
    sample_Per_perf_counters[i][INT_MUL32_ACC] = Per_imul32_accesses[i];
    sample_Per_perf_counters[i][INT_MUL_ACC] = Per_imul_accesses[i];
    sample_Per_perf_counters[i][INT_DIV_ACC] = Per_idiv_accesses[i];
  }
}

void gpgpu_sim_wrapper::set_dp_accesses(double dpu_accesses,
                                        double dpmul_accesses,
                                        double dpdiv_accesses) {
  sample_perf_counters[DP_ACC] = dpu_accesses;
  sample_perf_counters[DP_MUL_ACC] = dpmul_accesses;
  sample_perf_counters[DP_DIV_ACC] = dpdiv_accesses;
}

void gpgpu_sim_wrapper::set_Per_dp_accesses(const std::vector<double> &Per_dpu_accesses,
                                            const std::vector<double> &Per_dpmul_accesses,
                                            const std::vector<double> &Per_dpdiv_accesses) {
  for (unsigned i = 0; i < num_cores; i++) {
    sample_Per_perf_counters[i][DP_ACC] = Per_dpu_accesses[i];
    sample_Per_perf_counters[i][DP_MUL_ACC] = Per_dpmul_accesses[i];
    sample_Per_perf_counters[i][DP_DIV_ACC] = Per_dpdiv_accesses[i];
  }
}

void gpgpu_sim_wrapper::set_fp_accesses(double fpu_accesses,
                                        double fpmul_accesses,
                                        double fpdiv_accesses) {
  sample_perf_counters[FP_ACC] = fpu_accesses;
  sample_perf_counters[FP_MUL_ACC] = fpmul_accesses;
  sample_perf_counters[FP_DIV_ACC] = fpdiv_accesses;
}

void gpgpu_sim_wrapper::set_Per_fp_accesses(const std::vector<double> &Per_fpu_accesses,
                                            const std::vector<double> &Per_fpmul_accesses,
                                            const std::vector<double> &Per_fpdiv_accesses) {
  for (unsigned i = 0; i < num_cores; i++) {
    sample_Per_perf_counters[i][FP_ACC] = Per_fpu_accesses[i];
    sample_Per_perf_counters[i][FP_MUL_ACC] = Per_fpmul_accesses[i];
    sample_Per_perf_counters[i][FP_DIV_ACC] = Per_fpdiv_accesses[i];
  }
}

void gpgpu_sim_wrapper::set_trans_accesses(double sqrt_accesses,
                                           double log_accesses,
                                           double sin_accesses,
                                           double exp_accesses) {
  sample_perf_counters[FP_SQRT_ACC] = sqrt_accesses;
  sample_perf_counters[FP_LG_ACC] = log_accesses;
  sample_perf_counters[FP_SIN_ACC] = sin_accesses;
  sample_perf_counters[FP_EXP_ACC] = exp_accesses;
}

void gpgpu_sim_wrapper::set_Per_trans_accesses(const std::vector<double> &Per_sqrt_accesses,
                                               const std::vector<double> &Per_log_accesses,
                                               const std::vector<double> &Per_sin_accesses,
                                               const std::vector<double> &Per_exp_accesses) {
  for (unsigned i = 0; i < num_cores; i++) {
    sample_Per_perf_counters[i][FP_SQRT_ACC] = Per_sqrt_accesses[i];
    sample_Per_perf_counters[i][FP_LG_ACC] = Per_log_accesses[i];
    sample_Per_perf_counters[i][FP_SIN_ACC] = Per_sin_accesses[i];
    sample_Per_perf_counters[i][FP_EXP_ACC] = Per_exp_accesses[i];
  }
}

void gpgpu_sim_wrapper::set_tensor_accesses(double tensor_accesses) {
  sample_perf_counters[TENSOR_ACC] = tensor_accesses;
}

void gpgpu_sim_wrapper::set_tex_accesses(double tex_accesses) {
  sample_perf_counters[TEX_ACC] = tex_accesses;
}

void gpgpu_sim_wrapper::set_Per_tensor_tex_accesses(const std::vector<double> &Per_tensor_accesses,
                                                    const std::vector<double> &Per_tex_accesses) {
  for (unsigned i = 0; i < num_cores; i++) {
    sample_Per_perf_counters[i][TENSOR_ACC] = Per_tensor_accesses[i];
    sample_Per_perf_counters[i][TEX_ACC] = Per_tex_accesses[i];
  }
}

void gpgpu_sim_wrapper::set_avg_active_threads(float active_threads) {
  avg_threads_per_warp = (unsigned)ceil(active_threads);
  avg_threads_per_warp_tot += active_threads;
}

void gpgpu_sim_wrapper::set_active_lanes_power(double sp_avg_active_lane,
                                               double sfu_avg_active_lane) {
  p->sys.core[0].sp_average_active_lanes = sp_avg_active_lane;
  p->sys.core[0].sfu_average_active_lanes = sfu_avg_active_lane;
}
void gpgpu_sim_wrapper::set_Per_active_lanes_power(const std::vector<double> &Per_sp_avg_active_lane,
                                                const std::vector<double> &Per_sfu_avg_active_lane) {
  for (unsigned i = 0; i < num_cores; i++) {
    p->sys.core[i].sp_average_active_lanes = Per_sp_avg_active_lane[i];
    p->sys.core[i].sfu_average_active_lanes = Per_sfu_avg_active_lane[i];
  }
}

void gpgpu_sim_wrapper::set_NoC_power(double noc_tot_acc) {
  p->sys.NoC[0].total_accesses =
      noc_tot_acc * p->sys.scaling_coefficients[NOC_A];
  sample_perf_counters[NOC_A] = noc_tot_acc;
}

void gpgpu_sim_wrapper::power_metrics_calculations() {
  total_sample_count++;
  kernel_sample_count++;

  // Current sample power
  double sample_power = proc->rt_power.readOp.dynamic + sample_cmp_pwr[CONSTP] +
                        sample_cmp_pwr[STATICP];
  // double sample_power;
  // for(unsigned i=0; i<num_pwr_cmps; i++){
  //   sample_power+=sample_cmp_pwr[i]; //fix for dvfs
  // }

  // Average power
  // Previous + new + constant dynamic power (e.g., dynamic clocking power)
  kernel_tot_power += sample_power;
  kernel_power.avg = kernel_tot_power / kernel_sample_count;
  for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
    kernel_cmp_pwr[ind].avg += (double)sample_cmp_pwr[ind];
  }

  for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
    kernel_cmp_perf_counters[ind].avg += (double)sample_perf_counters[ind];
  }

  // Max Power
  if (sample_power > kernel_power.max) {
    kernel_power.max = sample_power;
    for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
      kernel_cmp_pwr[ind].max = (double)sample_cmp_pwr[ind];
    }
    for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
      kernel_cmp_perf_counters[ind].max = sample_perf_counters[ind];
    }
  }

  // Min Power
  if (sample_power < kernel_power.min || (kernel_power.min == 0)) {
    kernel_power.min = sample_power;
    for (unsigned ind = 0; ind < num_pwr_cmps; ++ind) {
      kernel_cmp_pwr[ind].min = (double)sample_cmp_pwr[ind];
    }
    for (unsigned ind = 0; ind < num_perf_counters; ++ind) {
      kernel_cmp_perf_counters[ind].min = sample_perf_counters[ind];
    }
  }

  gpu_tot_power.avg = (gpu_tot_power.avg + sample_power);
  gpu_tot_power.max =
      (sample_power > gpu_tot_power.max) ? sample_power : gpu_tot_power.max;
  gpu_tot_power.min =
      ((sample_power < gpu_tot_power.min) || (gpu_tot_power.min == 0))
          ? sample_power
          : gpu_tot_power.min;
  
}

void gpgpu_sim_wrapper::print_trace_files() {
  open_files();

  for (unsigned i = 0; i < num_perf_counters; ++i) {
    gzprintf(metric_trace_file, "%f,", sample_perf_counters[i]);
  }
  gzprintf(metric_trace_file, "\n");

  gzprintf(power_trace_file, "%f,", proc_power);
  for (unsigned i = 0; i < num_pwr_cmps; ++i) {
    gzprintf(power_trace_file, "%f,", sample_cmp_pwr[i]);
  }
  gzprintf(power_trace_file, "\n");

  close_files();
}

void gpgpu_sim_wrapper::update_coefficients() {
  std::vector<double> &init = initpower_coeff;
  std::vector<double> &eff  = effpower_coeff;
  double fpu_accesses = tot_fpu_accesses;
  double sfu_accesses = tot_sfu_accesses;

  init[FP_INT] = proc->cores[0]->get_coefficient_fpint_insts();
  eff[FP_INT] =
      init[FP_INT] * p->sys.scaling_coefficients[FP_INT];

  init[TOT_INST] = proc->cores[0]->get_coefficient_tot_insts();
  eff[TOT_INST] =
      init[TOT_INST] * p->sys.scaling_coefficients[TOT_INST];

  init[REG_RD] =
      proc->cores[0]->get_coefficient_regreads_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  init[REG_WR] =
      proc->cores[0]->get_coefficient_regwrites_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  init[NON_REG_OPs] =
      proc->cores[0]->get_coefficient_noregfileops_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
  eff[REG_RD] =
      init[REG_RD] * p->sys.scaling_coefficients[REG_RD];
  eff[REG_WR] =
      init[REG_WR] * p->sys.scaling_coefficients[REG_WR];
  eff[NON_REG_OPs] =
      init[NON_REG_OPs] * p->sys.scaling_coefficients[NON_REG_OPs];

  init[IC_H] = proc->cores[0]->get_coefficient_icache_hits();
  init[IC_M] = proc->cores[0]->get_coefficient_icache_misses();
  eff[IC_H] =
      init[IC_H] * p->sys.scaling_coefficients[IC_H];
  eff[IC_M] =
      init[IC_M] * p->sys.scaling_coefficients[IC_M];

  init[CC_H] = (proc->cores[0]->get_coefficient_ccache_readhits() +
                           proc->get_coefficient_readcoalescing());
  init[CC_M] = (proc->cores[0]->get_coefficient_ccache_readmisses() +
                           proc->get_coefficient_readcoalescing());
  eff[CC_H] =
      init[CC_H] * p->sys.scaling_coefficients[CC_H];
  eff[CC_M] =
      init[CC_M] * p->sys.scaling_coefficients[CC_M];

  init[TC_H] = (proc->cores[0]->get_coefficient_tcache_readhits() +
                           proc->get_coefficient_readcoalescing());
  init[TC_M] = (proc->cores[0]->get_coefficient_tcache_readmisses() +
                           proc->get_coefficient_readcoalescing());
  eff[TC_H] =
      init[TC_H] * p->sys.scaling_coefficients[TC_H];
  eff[TC_M] =
      init[TC_M] * p->sys.scaling_coefficients[TC_M];

  init[SHRD_ACC] =
      proc->cores[0]->get_coefficient_sharedmemory_readhits();
  eff[SHRD_ACC] =
      init[SHRD_ACC] * p->sys.scaling_coefficients[SHRD_ACC];

  init[DC_RH] = (proc->cores[0]->get_coefficient_dcache_readhits() +
                            proc->get_coefficient_readcoalescing());
  init[DC_RM] =
      (proc->cores[0]->get_coefficient_dcache_readmisses() +
       proc->get_coefficient_readcoalescing());
  init[DC_WH] = (proc->cores[0]->get_coefficient_dcache_writehits() +
                            proc->get_coefficient_writecoalescing());
  init[DC_WM] =
      (proc->cores[0]->get_coefficient_dcache_writemisses() +
       proc->get_coefficient_writecoalescing());
  eff[DC_RH] =
      init[DC_RH] * p->sys.scaling_coefficients[DC_RH];
  eff[DC_RM] =
      init[DC_RM] * p->sys.scaling_coefficients[DC_RM];
  eff[DC_WH] =
      init[DC_WH] * p->sys.scaling_coefficients[DC_WH];
  eff[DC_WM] =
      init[DC_WM] * p->sys.scaling_coefficients[DC_WM];

  init[L2_RH] = proc->get_coefficient_l2_read_hits();
  init[L2_RM] = proc->get_coefficient_l2_read_misses();
  init[L2_WH] = proc->get_coefficient_l2_write_hits();
  init[L2_WM] = proc->get_coefficient_l2_write_misses();
  eff[L2_RH] =
      init[L2_RH] * p->sys.scaling_coefficients[L2_RH];
  eff[L2_RM] =
      init[L2_RM] * p->sys.scaling_coefficients[L2_RM];
  eff[L2_WH] =
      init[L2_WH] * p->sys.scaling_coefficients[L2_WH];
  eff[L2_WM] =
      init[L2_WM] * p->sys.scaling_coefficients[L2_WM];

  init[IDLE_CORE_N] =
      p->sys.idle_core_power * proc->cores[0]->executionTime;
  eff[IDLE_CORE_N] =
      init[IDLE_CORE_N] * p->sys.scaling_coefficients[IDLE_CORE_N];

  init[PIPE_A] = proc->cores[0]->get_coefficient_duty_cycle();
  eff[PIPE_A] =
      init[PIPE_A] * p->sys.scaling_coefficients[PIPE_A];

  init[MEM_RD] = proc->get_coefficient_mem_reads();
  init[MEM_WR] = proc->get_coefficient_mem_writes();
  init[MEM_PRE] = proc->get_coefficient_mem_pre();
  eff[MEM_RD] =
      init[MEM_RD] * p->sys.scaling_coefficients[MEM_RD];
  eff[MEM_WR] =
      init[MEM_WR] * p->sys.scaling_coefficients[MEM_WR];
  eff[MEM_PRE] =
      init[MEM_PRE] * p->sys.scaling_coefficients[MEM_PRE];

  double fp_coeff = proc->cores[0]->get_coefficient_fpu_accesses();
  double sfu_coeff = proc->cores[0]->get_coefficient_sfu_accesses();

  init[INT_ACC] =
      proc->cores[0]->get_coefficient_ialu_accesses() *
      (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);

  if (fpu_accesses != 0) {
    init[FP_ACC] =
        fp_coeff * sample_perf_counters[FP_ACC] / fpu_accesses;
    init[DP_ACC] =
        fp_coeff * sample_perf_counters[DP_ACC] / fpu_accesses;
  } else {
    init[FP_ACC] = 0;
    init[DP_ACC] = 0;
  }

  if (sfu_accesses != 0) {
    init[INT_MUL24_ACC] =
        sfu_coeff * sample_perf_counters[INT_MUL24_ACC] / sfu_accesses;
    init[INT_MUL32_ACC] =
        sfu_coeff * sample_perf_counters[INT_MUL32_ACC] / sfu_accesses;
    init[INT_MUL_ACC] =
        sfu_coeff * sample_perf_counters[INT_MUL_ACC] / sfu_accesses;
    init[INT_DIV_ACC] =
        sfu_coeff * sample_perf_counters[INT_DIV_ACC] / sfu_accesses;
    init[DP_MUL_ACC] =
        sfu_coeff * sample_perf_counters[DP_MUL_ACC] / sfu_accesses;
    init[DP_DIV_ACC] =
        sfu_coeff * sample_perf_counters[DP_DIV_ACC] / sfu_accesses;
    init[FP_MUL_ACC] =
        sfu_coeff * sample_perf_counters[FP_MUL_ACC] / sfu_accesses;
    init[FP_DIV_ACC] =
        sfu_coeff * sample_perf_counters[FP_DIV_ACC] / sfu_accesses;
    init[FP_SQRT_ACC] =
        sfu_coeff * sample_perf_counters[FP_SQRT_ACC] / sfu_accesses;
    init[FP_LG_ACC] =
        sfu_coeff * sample_perf_counters[FP_LG_ACC] / sfu_accesses;
    init[FP_SIN_ACC] =
        sfu_coeff * sample_perf_counters[FP_SIN_ACC] / sfu_accesses;
    init[FP_EXP_ACC] =
        sfu_coeff * sample_perf_counters[FP_EXP_ACC] / sfu_accesses;
    init[TENSOR_ACC] =
        sfu_coeff * sample_perf_counters[TENSOR_ACC] / sfu_accesses;
    init[TEX_ACC] =
        sfu_coeff * sample_perf_counters[TEX_ACC] / sfu_accesses;
  } else {
    init[INT_MUL24_ACC] = 0;
    init[INT_MUL32_ACC] = 0;
    init[INT_MUL_ACC] = 0;
    init[INT_DIV_ACC] = 0;
    init[DP_MUL_ACC] = 0;
    init[DP_DIV_ACC] = 0;
    init[FP_MUL_ACC] = 0;
    init[FP_DIV_ACC] = 0;
    init[FP_SQRT_ACC] = 0;
    init[FP_LG_ACC] = 0;
    init[FP_SIN_ACC] = 0;
    init[FP_EXP_ACC] = 0;
    init[TENSOR_ACC] = 0;
    init[TEX_ACC] = 0;
  }

  eff[INT_ACC] = init[INT_ACC];
  eff[FP_ACC] = init[FP_ACC];
  eff[DP_ACC] = init[DP_ACC];
  eff[INT_MUL24_ACC] = init[INT_MUL24_ACC];
  eff[INT_MUL32_ACC] = init[INT_MUL32_ACC];
  eff[INT_MUL_ACC] = init[INT_MUL_ACC];
  eff[INT_DIV_ACC] = init[INT_DIV_ACC];
  eff[DP_MUL_ACC] = init[DP_MUL_ACC];
  eff[DP_DIV_ACC] = init[DP_DIV_ACC];
  eff[FP_MUL_ACC] = init[FP_MUL_ACC];
  eff[FP_DIV_ACC] = init[FP_DIV_ACC];
  eff[FP_SQRT_ACC] = init[FP_SQRT_ACC];
  eff[FP_LG_ACC] = init[FP_LG_ACC];
  eff[FP_SIN_ACC] = init[FP_SIN_ACC];
  eff[FP_EXP_ACC] = init[FP_EXP_ACC];
  eff[TENSOR_ACC] = init[TENSOR_ACC];
  eff[TEX_ACC] = init[TEX_ACC];

  init[NOC_A] = proc->get_coefficient_noc_accesses();
  eff[NOC_A] =
      init[NOC_A] * p->sys.scaling_coefficients[NOC_A];

  // const_dynamic_power=proc->get_const_dynamic_power()/(proc->cores[0]->executionTime);

  for (unsigned i = 0; i < num_perf_counters; i++) {
    init[i] /= (proc->cores[0]->executionTime);
    eff[i] /= (proc->cores[0]->executionTime);
  }
}

double gpgpu_sim_wrapper::calculate_static_power() {
  std::vector<double> &init = initpower_coeff;
  std::vector<double> &eff  = effpower_coeff;
  double int_accesses =
      init[INT_ACC] + init[INT_MUL24_ACC] +
      init[INT_MUL32_ACC] + init[INT_MUL_ACC] +
      init[INT_DIV_ACC];
  double int_add_accesses = init[INT_ACC];
  double int_mul_accesses =
      init[INT_MUL24_ACC] + init[INT_MUL32_ACC] +
      init[INT_MUL_ACC] + init[INT_DIV_ACC];
  double fp_accesses = init[FP_ACC] + init[FP_MUL_ACC] +
                       init[FP_DIV_ACC];
  double dp_accesses = init[DP_ACC] + init[DP_MUL_ACC] +
                       init[DP_DIV_ACC];
  double sfu_accesses =
      init[FP_SQRT_ACC] + init[FP_LG_ACC] +
      init[FP_SIN_ACC] + init[FP_EXP_ACC];
  double tensor_accesses = init[TENSOR_ACC];
  double tex_accesses = init[TEX_ACC];
  double total_static_power = 0.0;
  double base_static_power = 0.0;
  double lane_static_power = 0.0;
  double per_active_core = (num_cores - num_idle_cores) / num_cores;
  // lyhong_file << "lyhong_print:" << " num_cores: " << num_cores << " num_idle_cores: " << num_idle_cores << " per_active_core: " << per_active_core << std::endl;

  double l1_accesses = init[DC_RH] + init[DC_RM] +
                       init[DC_WH] + init[DC_WM];
  double l2_accesses = init[L2_RH] + init[L2_RM] +
                       init[L2_WH] + init[L2_WM];
  double shared_accesses = init[SHRD_ACC];

  if (avg_threads_per_warp ==
      0) {  // no functional unit threads, check for memory or a 'LIGHT_SM'
    if (l1_accesses != 0.0)
      return (p->sys.static_l1_flane * per_active_core);
    else if (shared_accesses != 0.0)
      return (p->sys.static_shared_flane * per_active_core);
    else if (l2_accesses != 0.0)
      return (p->sys.static_l2_flane * per_active_core);
    else  // LIGHT_SM
      return (p->sys.static_light_flane *
              per_active_core);  // return LIGHT_SM base static power
  }

  /* using a linear model for thread divergence */
  if ((int_accesses != 0.0) && (fp_accesses != 0.0) && (dp_accesses != 0.0) &&
      (sfu_accesses == 0.0) && (tensor_accesses == 0.0) &&
      (tex_accesses == 0.0)) {
    /* INT_FP_DP */
    base_static_power = p->sys.static_cat3_flane;
    lane_static_power = p->sys.static_cat3_addlane;
  }

  else if ((int_accesses != 0.0) && (fp_accesses != 0.0) &&
           (dp_accesses == 0.0) && (sfu_accesses == 0.0) &&
           (tensor_accesses != 0.0) && (tex_accesses == 0.0)) {
    /* INT_FP_TENSOR */
    base_static_power = p->sys.static_cat6_flane;
    lane_static_power = p->sys.static_cat6_addlane;
  }

  else if ((int_accesses != 0.0) && (fp_accesses != 0.0) &&
           (dp_accesses == 0.0) && (sfu_accesses != 0.0) &&
           (tensor_accesses == 0.0) && (tex_accesses == 0.0)) {
    /* INT_FP_SFU */
    base_static_power = p->sys.static_cat4_flane;
    lane_static_power = p->sys.static_cat4_addlane;
  }

  else if ((int_accesses != 0.0) && (fp_accesses != 0.0) &&
           (dp_accesses == 0.0) && (sfu_accesses == 0.0) &&
           (tensor_accesses == 0.0) && (tex_accesses != 0.0)) {
    /* INT_FP_TEX */
    base_static_power = p->sys.static_cat5_flane;
    lane_static_power = p->sys.static_cat5_addlane;
  }

  else if ((int_accesses != 0.0) && (fp_accesses != 0.0) &&
           (dp_accesses == 0.0) && (sfu_accesses == 0.0) &&
           (tensor_accesses == 0.0) && (tex_accesses == 0.0)) {
    /* INT_FP */
    base_static_power = p->sys.static_cat2_flane;
    lane_static_power = p->sys.static_cat2_addlane;
  }

  else if ((int_accesses != 0.0) && (fp_accesses == 0.0) &&
           (dp_accesses == 0.0) && (sfu_accesses == 0.0) &&
           (tensor_accesses == 0.0) && (tex_accesses == 0.0)) {
    /* INT */
    /* Seperating INT_ADD only and INT_MUL only from mix of INT instructions */
    if ((int_add_accesses != 0.0) && (int_mul_accesses == 0.0)) {  // INT_ADD
      base_static_power = p->sys.static_intadd_flane;
      lane_static_power = p->sys.static_intadd_addlane;
    } else if ((int_add_accesses == 0.0) &&
               (int_mul_accesses != 0.0)) {  // INT_MUL
      base_static_power = p->sys.static_intmul_flane;
      lane_static_power = p->sys.static_intmul_addlane;
    } else {  // INT_ADD+MUL
      base_static_power = p->sys.static_cat1_flane;
      lane_static_power = p->sys.static_cat1_addlane;
    }
  }

  else if ((int_accesses == 0.0) && (fp_accesses == 0.0) &&
           (dp_accesses == 0.0) && (sfu_accesses == 0.0) &&
           (tensor_accesses == 0.0) && (tex_accesses == 0.0)) {
    /* LIGHT_SM or memory only sample */
    lane_static_power =
        0.0;  // addlane static power is 0 for l1/l2/shared memory only accesses
    if (l1_accesses != 0.0)
      base_static_power = p->sys.static_l1_flane;
    else if (shared_accesses != 0.0)
      base_static_power = p->sys.static_shared_flane;
    else if (l2_accesses != 0.0)
      base_static_power = p->sys.static_l2_flane;
    else {
      base_static_power = p->sys.static_light_flane;
      lane_static_power = p->sys.static_light_addlane;
    }
  } else {
    base_static_power =
        p->sys.static_geomean_flane;  // GEOMEAN except LIGHT_SM if we don't
                                      // fall into any of the categories above
    lane_static_power = p->sys.static_geomean_addlane;
  }

  total_static_power =
      base_static_power + (((double)avg_threads_per_warp - 1.0) *
                           lane_static_power);  // Linear Model
  return (total_static_power * per_active_core);
}

void gpgpu_sim_wrapper::update_components_power() {
  if(!Lyhong_Percore_sim) {
    update_coefficients();
    proc_power = proc->rt_power.readOp.dynamic;
    sample_cmp_pwr[IBP] =
        (proc->cores[0]->ifu->IB->rt_power.readOp.dynamic +
        proc->cores[0]->ifu->IB->rt_power.writeOp.dynamic +
        proc->cores[0]->ifu->ID_misc->rt_power.readOp.dynamic +
        proc->cores[0]->ifu->ID_operand->rt_power.readOp.dynamic +
        proc->cores[0]->ifu->ID_inst->rt_power.readOp.dynamic) /
        (proc->cores[0]->executionTime);

    sample_cmp_pwr[ICP] = proc->cores[0]->ifu->icache.rt_power.readOp.dynamic /
                          (proc->cores[0]->executionTime);

    sample_cmp_pwr[DCP] = proc->cores[0]->lsu->dcache.rt_power.readOp.dynamic /
                          (proc->cores[0]->executionTime);

    sample_cmp_pwr[TCP] = proc->cores[0]->lsu->tcache.rt_power.readOp.dynamic /
                          (proc->cores[0]->executionTime);

    sample_cmp_pwr[CCP] = proc->cores[0]->lsu->ccache.rt_power.readOp.dynamic /
                          (proc->cores[0]->executionTime);

    sample_cmp_pwr[SHRDP] =
        proc->cores[0]->lsu->sharedmemory.rt_power.readOp.dynamic /
        (proc->cores[0]->executionTime);

    sample_cmp_pwr[RFP] =
        (proc->cores[0]->exu->rfu->rt_power.readOp.dynamic /
        (proc->cores[0]->executionTime)) *
        (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);

    double sample_fp_pwr = (proc->cores[0]->exu->fp_u->rt_power.readOp.dynamic /
                            (proc->cores[0]->executionTime));

    double sample_sfu_pwr = (proc->cores[0]->exu->mul->rt_power.readOp.dynamic /
                            (proc->cores[0]->executionTime));

    sample_cmp_pwr[INTP] =
        (proc->cores[0]->exu->exeu->rt_power.readOp.dynamic /
        (proc->cores[0]->executionTime)) *
        (proc->cores[0]->exu->rf_fu_clockRate / proc->cores[0]->exu->clockRate);
    if (tot_fpu_accesses != 0) {
      sample_cmp_pwr[FPUP] =
          sample_fp_pwr * sample_perf_counters[FP_ACC] / tot_fpu_accesses;
      sample_cmp_pwr[DPUP] =
          sample_fp_pwr * sample_perf_counters[DP_ACC] / tot_fpu_accesses;
    } else {
      sample_cmp_pwr[FPUP] = 0;
      sample_cmp_pwr[DPUP] = 0;
    }
    if (tot_sfu_accesses != 0) {
      sample_cmp_pwr[INT_MUL24P] =
          sample_sfu_pwr * sample_perf_counters[INT_MUL24_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[INT_MUL32P] =
          sample_sfu_pwr * sample_perf_counters[INT_MUL32_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[INT_MULP] =
          sample_sfu_pwr * sample_perf_counters[INT_MUL_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[INT_DIVP] =
          sample_sfu_pwr * sample_perf_counters[INT_DIV_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[FP_MULP] =
          sample_sfu_pwr * sample_perf_counters[FP_MUL_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[FP_DIVP] =
          sample_sfu_pwr * sample_perf_counters[FP_DIV_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[FP_SQRTP] =
          sample_sfu_pwr * sample_perf_counters[FP_SQRT_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[FP_LGP] =
          sample_sfu_pwr * sample_perf_counters[FP_LG_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[FP_SINP] =
          sample_sfu_pwr * sample_perf_counters[FP_SIN_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[FP_EXP] =
          sample_sfu_pwr * sample_perf_counters[FP_EXP_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[DP_MULP] =
          sample_sfu_pwr * sample_perf_counters[DP_MUL_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[DP_DIVP] =
          sample_sfu_pwr * sample_perf_counters[DP_DIV_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[TENSORP] =
          sample_sfu_pwr * sample_perf_counters[TENSOR_ACC] / tot_sfu_accesses;
      sample_cmp_pwr[TEXP] =
          sample_sfu_pwr * sample_perf_counters[TEX_ACC] / tot_sfu_accesses;
    } else {
    sample_cmp_pwr[INT_MUL24P] = 0;
    sample_cmp_pwr[INT_MUL32P] = 0;
    sample_cmp_pwr[INT_MULP] = 0;
    sample_cmp_pwr[INT_DIVP] = 0;
    sample_cmp_pwr[FP_MULP] = 0;
    sample_cmp_pwr[FP_DIVP] = 0;
    sample_cmp_pwr[FP_SQRTP] = 0;
    sample_cmp_pwr[FP_LGP] = 0;
    sample_cmp_pwr[FP_SINP] = 0;
    sample_cmp_pwr[FP_EXP] = 0;
    sample_cmp_pwr[DP_MULP] = 0;
    sample_cmp_pwr[DP_DIVP] = 0;
    sample_cmp_pwr[TENSORP] = 0;
    sample_cmp_pwr[TEXP] = 0;
  }
    sample_cmp_pwr[SCHEDP] = proc->cores[0]->exu->scheu->rt_power.readOp.dynamic /
                            (proc->cores[0]->executionTime);

    sample_cmp_pwr[L2CP] = (proc->XML->sys.number_of_L2s > 0)
                              ? proc->l2array[0]->rt_power.readOp.dynamic /
                                    (proc->cores[0]->executionTime)
                              : 0;

    sample_cmp_pwr[MCP] = (proc->mc->rt_power.readOp.dynamic -
                          proc->mc->dram->rt_power.readOp.dynamic) /
                          (proc->cores[0]->executionTime);

    sample_cmp_pwr[NOCP] =
        proc->nocs[0]->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);

    sample_cmp_pwr[DRAMP] =
        proc->mc->dram->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);

    sample_cmp_pwr[PIPEP] =
        proc->cores[0]->Pipeline_energy / (proc->cores[0]->executionTime);

    sample_cmp_pwr[IDLE_COREP] =
        proc->cores[0]->IdleCoreEnergy / (proc->cores[0]->executionTime);

    // This constant dynamic power (e.g., clock power) part is estimated via
    // regression model.
    sample_cmp_pwr[CONSTP] = 0;
    sample_cmp_pwr[STATICP] = 0;
    // double cnst_dyn =
    // proc->get_const_dynamic_power()/(proc->cores[0]->executionTime);
    // // If the regression scaling term is greater than the recorded constant
    // dynamic power
    // // then use the difference (other portion already added to dynamic power).
    // Else,
    // // all the constant dynamic power is accounted for, add nothing.
    // if(p->sys.scaling_coefficients[constant_power] > cnst_dyn)
    //   sample_cmp_pwr[CONSTP] =
    //   (p->sys.scaling_coefficients[constant_power]-cnst_dyn);
    sample_cmp_pwr[CONSTP] = p->sys.scaling_coefficients[constant_power];
    sample_cmp_pwr[STATICP] = calculate_static_power();

    if (g_dvfs_enabled) {
      double voltage_ratio =
          modeled_chip_voltage / p->sys.modeled_chip_voltage_ref;
      sample_cmp_pwr[IDLE_COREP] *=
          voltage_ratio;  // static power scaled by voltage_ratio
      sample_cmp_pwr[STATICP] *=
          voltage_ratio;  // static power scaled by voltage_ratio
      for (unsigned i = 0; i < num_pwr_cmps; i++) {
        if ((i != IDLE_COREP) && (i != STATICP)) {
          sample_cmp_pwr[i] *=
              voltage_ratio *
              voltage_ratio;  // dynamic power scaled by square of voltage_ratio
        }
      }
    }

    proc_power += sample_cmp_pwr[CONSTP] + sample_cmp_pwr[STATICP];
    if (!g_dvfs_enabled) {  // sanity check will fail when voltage scaling is
                            // applied, fix later
      double sum_pwr_cmp = 0;
      for (unsigned i = 0; i < num_pwr_cmps; i++) {
        sum_pwr_cmp += sample_cmp_pwr[i];
      }
      bool check = false;
      check = sanity_check(sum_pwr_cmp, proc_power);
      if (!check)
        printf("sum_pwr_cmp %f : proc_power %f \n", sum_pwr_cmp, proc_power);
      assert("Total Power does not equal the sum of the components\n" && (check));
    }
  } else {  // Per SM power
    update_coefficients();
    for (unsigned i = 0; i < num_cores; i++) {
      proc_power = proc->rt_power.readOp.dynamic;
      sample_Per_cmp_pwr[i][IBP] =
          (proc->cores[i]->ifu->IB->rt_power.readOp.dynamic +
          proc->cores[i]->ifu->IB->rt_power.writeOp.dynamic +
          proc->cores[i]->ifu->ID_misc->rt_power.readOp.dynamic +
          proc->cores[i]->ifu->ID_operand->rt_power.readOp.dynamic +
          proc->cores[i]->ifu->ID_inst->rt_power.readOp.dynamic) /
          (proc->cores[i]->executionTime);

      sample_Per_cmp_pwr[i][ICP] = proc->cores[i]->ifu->icache.rt_power.readOp.dynamic /
                            (proc->cores[i]->executionTime);

      sample_Per_cmp_pwr[i][DCP] = proc->cores[i]->lsu->dcache.rt_power.readOp.dynamic /
                            (proc->cores[i]->executionTime);

      sample_Per_cmp_pwr[i][TCP] = proc->cores[i]->lsu->tcache.rt_power.readOp.dynamic /
                            (proc->cores[i]->executionTime);

      sample_Per_cmp_pwr[i][CCP] = proc->cores[i]->lsu->ccache.rt_power.readOp.dynamic /
                            (proc->cores[i]->executionTime);

      sample_Per_cmp_pwr[i][SHRDP] =
          proc->cores[i]->lsu->sharedmemory.rt_power.readOp.dynamic /
          (proc->cores[i]->executionTime);

      sample_Per_cmp_pwr[i][RFP] =
          (proc->cores[i]->exu->rfu->rt_power.readOp.dynamic /
          (proc->cores[i]->executionTime)) *
          (proc->cores[i]->exu->rf_fu_clockRate / proc->cores[i]->exu->clockRate);

      double sample_fp_pwr = (proc->cores[i]->exu->fp_u->rt_power.readOp.dynamic /
                              (proc->cores[i]->executionTime));

      double sample_sfu_pwr = (proc->cores[i]->exu->mul->rt_power.readOp.dynamic /
                              (proc->cores[i]->executionTime));

      sample_Per_cmp_pwr[i][INTP] =
          (proc->cores[i]->exu->exeu->rt_power.readOp.dynamic /
          (proc->cores[i]->executionTime)) *
          (proc->cores[i]->exu->rf_fu_clockRate / proc->cores[i]->exu->clockRate);
      if (tot_fpu_accesses != 0) {
        sample_Per_cmp_pwr[i][FPUP] =
          sample_fp_pwr * sample_Per_perf_counters[i][FP_ACC] / tot_fpu_accesses;
        sample_Per_cmp_pwr[i][DPUP] =
          sample_fp_pwr * sample_Per_perf_counters[i][DP_ACC] / tot_fpu_accesses;
      } else {
        sample_Per_cmp_pwr[i][FPUP] = 0;
        sample_Per_cmp_pwr[i][DPUP] = 0;
      }
      if (tot_sfu_accesses != 0) {
        sample_Per_cmp_pwr[i][INT_MUL24P] =
          sample_sfu_pwr * sample_Per_perf_counters[i][INT_MUL24_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][INT_MUL32P] =
          sample_sfu_pwr * sample_Per_perf_counters[i][INT_MUL32_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][INT_DIVP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][INT_DIV_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][FP_MULP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][FP_MUL_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][FP_DIVP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][FP_DIV_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][FP_SQRTP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][FP_SQRT_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][FP_LGP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][FP_LG_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][FP_SINP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][FP_SIN_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][FP_EXP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][FP_EXP_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][DP_MULP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][DP_MUL_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][DP_DIVP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][DP_DIV_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][TENSORP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][TENSOR_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][TEXP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][TEX_ACC] / tot_sfu_accesses;
        sample_Per_cmp_pwr[i][INT_MULP] =
          sample_sfu_pwr * sample_Per_perf_counters[i][INT_MUL_ACC] / tot_sfu_accesses;
      } else {
        sample_Per_cmp_pwr[i][INT_MUL24P] = 0;
        sample_Per_cmp_pwr[i][INT_MUL32P] = 0;
        sample_Per_cmp_pwr[i][INT_DIVP] = 0;
        sample_Per_cmp_pwr[i][FP_MULP] = 0;
        sample_Per_cmp_pwr[i][FP_DIVP] = 0;
        sample_Per_cmp_pwr[i][FP_SQRTP] = 0;
        sample_Per_cmp_pwr[i][FP_LGP] = 0;
        sample_Per_cmp_pwr[i][FP_SINP] = 0;
        sample_Per_cmp_pwr[i][FP_EXP] = 0;
        sample_Per_cmp_pwr[i][DP_MULP] = 0;
        sample_Per_cmp_pwr[i][DP_DIVP] = 0;
        sample_Per_cmp_pwr[i][TENSORP] = 0;
        sample_Per_cmp_pwr[i][TEXP] = 0;
        sample_Per_cmp_pwr[i][INT_MULP] = 0;
      }
      sample_Per_cmp_pwr[i][SCHEDP] = proc->cores[i]->exu->scheu->rt_power.readOp.dynamic /
                          (proc->cores[i]->executionTime);

      sample_Per_cmp_pwr[i][PIPEP] =
          proc->cores[i]->Pipeline_energy / (proc->cores[i]->executionTime);
    }
    sample_cmp_pwr[L2CP] = (proc->XML->sys.number_of_L2s > 0)
                            ? proc->l2array[0]->rt_power.readOp.dynamic /
                                  (proc->cores[0]->executionTime)
                            : 0;

    sample_cmp_pwr[MCP] = (proc->mc->rt_power.readOp.dynamic -
                          proc->mc->dram->rt_power.readOp.dynamic) /
                          (proc->cores[0]->executionTime);

    sample_cmp_pwr[NOCP] =
        proc->nocs[0]->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);
    
    sample_cmp_pwr[DRAMP] =
        proc->mc->dram->rt_power.readOp.dynamic / (proc->cores[0]->executionTime);

    sample_cmp_pwr[CONSTP] = 0;
    sample_cmp_pwr[STATICP] = 0;
    sample_cmp_pwr[CONSTP] = p->sys.scaling_coefficients[constant_power];
    sample_cmp_pwr[STATICP] = calculate_static_power();
    if (g_dvfs_enabled) {
      double voltage_ratio =
          modeled_chip_voltage / p->sys.modeled_chip_voltage_ref;
      sample_cmp_pwr[IDLE_COREP] *=
          voltage_ratio;  // static power scaled by voltage_ratio
      sample_cmp_pwr[STATICP] *=
          voltage_ratio;  // static power scaled by voltage_ratio
      for (unsigned i = 0; i < num_pwr_cmps; i++) {
        if ((i != IDLE_COREP) && (i != STATICP)) {
          sample_cmp_pwr[i] *=
              voltage_ratio *
              voltage_ratio;  // dynamic power scaled by square of voltage_ratio
        }
      }
    }

    proc_power += sample_cmp_pwr[CONSTP] + sample_cmp_pwr[STATICP];
    // if (!g_dvfs_enabled) {  // sanity check will fail when voltage scaling is
    //                         // applied, fix later
    //   double sum_pwr_cmp = 0;
    //   for (unsigned i = 0; i < num_pwr_cmps; i++) {
    //     sum_pwr_cmp += sample_cmp_pwr[i];
    //   }
    //   bool check = false;
    //   check = sanity_check(sum_pwr_cmp, proc_power);
    //   if (!check)
    //     printf("sum_pwr_cmp %f : proc_power %f \n", sum_pwr_cmp, proc_power);
    //   assert("Total Power does not equal the sum of the components\n" && (check));
    // }

    if (!sm_header_dumped) {
      for (unsigned i = 0; i < num_cores; i++) {
        lyhong_file << "SM_" << i << "\t";
      }
      lyhong_file << "Total_Power" << std::endl;
      if(Percore_detail) {
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_INT_MULP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_INT_MUL24P\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_INT_MUL32P\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_INT_DIVP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_FP_DIVP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_FP_SQRTP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_FP_LGP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_FP_SINP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_DP_DIVP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_TENSORP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_TEXP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_DPUP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_DP_MULP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_FP_MULP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_FP_EXP\t";
        for (unsigned i = 0; i < num_cores; i++)
          lyhong_SM_file << "SM_" << i << "_FPUP" << (i + 1 < num_cores ? "\t" : "");
        lyhong_SM_file << std::endl;
      }
      sm_header_dumped = true;
    }
    double Total_Power_SM = 0.0;
    for (unsigned i = 0; i < num_cores; i++) {
      SM_core_power[i] = sample_Per_cmp_pwr[i][IBP] + 
                          sample_Per_cmp_pwr[i][ICP] + 
                          sample_Per_cmp_pwr[i][DCP] + 
                          sample_Per_cmp_pwr[i][TCP] + 
                          sample_Per_cmp_pwr[i][SHRDP] + 
                          sample_Per_cmp_pwr[i][RFP] + 
                          sample_Per_cmp_pwr[i][INTP] + 
                          sample_Per_cmp_pwr[i][FPUP] + 
                          sample_Per_cmp_pwr[i][DPUP] + 
                          sample_Per_cmp_pwr[i][INT_MULP] + 
                          sample_Per_cmp_pwr[i][INT_MUL24P] + 
                          sample_Per_cmp_pwr[i][INT_MUL32P] + 
                          sample_Per_cmp_pwr[i][INT_DIVP] + 
                          sample_Per_cmp_pwr[i][FP_MULP] + 
                          sample_Per_cmp_pwr[i][FP_DIVP] + 
                          sample_Per_cmp_pwr[i][FP_SQRTP] + 
                          sample_Per_cmp_pwr[i][FP_SINP] + 
                          sample_Per_cmp_pwr[i][FP_LGP] + 
                          sample_Per_cmp_pwr[i][FP_EXP] + 
                          sample_Per_cmp_pwr[i][DP_MULP] + 
                          sample_Per_cmp_pwr[i][DP_DIVP] + 
                          sample_Per_cmp_pwr[i][TENSORP] + 
                          sample_Per_cmp_pwr[i][TEXP] + 
                          sample_Per_cmp_pwr[i][SCHEDP] + 
                          sample_Per_cmp_pwr[i][IDLE_COREP] + 
                          sample_Per_cmp_pwr[i][PIPEP];
      Total_Power_SM += SM_core_power[i];
      lyhong_file << SM_core_power[i] << "\t";
    }
    lyhong_file << Total_Power_SM << std::endl;
    // lyhong_file << "IBP: " << sample_Per_cmp_pwr[0][IBP] << std::endl;
    // lyhong_file << "ICP: " << sample_Per_cmp_pwr[0][ICP] << std::endl;
    // lyhong_file << "DCP: " << sample_Per_cmp_pwr[0][DCP] << std::endl;
    // lyhong_file << "TCP: " << sample_Per_cmp_pwr[0][TCP] << std::endl;
    // lyhong_file << "SHRDP: " << sample_Per_cmp_pwr[0][SHRDP] << std::endl;
    // lyhong_file << "RFP: " << sample_Per_cmp_pwr[0][RFP] << std::endl;
    // lyhong_file << "INTP: " << sample_Per_cmp_pwr[0][INTP] << std::endl;
    // lyhong_file << "FPUP: " << sample_Per_cmp_pwr[0][FPUP] << std::endl;
    // lyhong_file << "DPUP: " << sample_Per_cmp_pwr[0][DPUP] << std::endl;
    // lyhong_file << "INT_MULP: " << sample_Per_cmp_pwr[0][INT_MULP] << std::endl;
    // lyhong_file << "INT_MUL24P: " << sample_Per_cmp_pwr[0][INT_MUL24P] << std::endl;
    // lyhong_file << "INT_MUL32P: " << sample_Per_cmp_pwr[0][INT_MUL32P] << std::endl;
    // lyhong_file << "INT_DIVP: " << sample_Per_cmp_pwr[0][INT_DIVP] << std::endl;
    // lyhong_file << "FP_MULP: " << sample_Per_cmp_pwr[0][FP_MULP] << std::endl;
    // lyhong_file << "FP_DIVP: " << sample_Per_cmp_pwr[0][FP_DIVP] << std::endl;
    // lyhong_file << "FP_SQRTP: " << sample_Per_cmp_pwr[0][FP_SQRTP] << std::endl;
    // lyhong_file << "FP_LGP: " << sample_Per_cmp_pwr[0][FP_LGP] << std::endl;
    // lyhong_file << "FP_SINP: " << sample_Per_cmp_pwr[0][FP_SINP] << std::endl;
    // lyhong_file << "FP_EXP: " << sample_Per_cmp_pwr[0][FP_EXP] << std::endl;
    // lyhong_file << "DP_MULP: " << sample_Per_cmp_pwr[0][DP_MULP] << std::endl;
    // lyhong_file << "DP_DIVP: " << sample_Per_cmp_pwr[0][DP_DIVP] << std::endl;
    // lyhong_file << "TENSORP: " << sample_Per_cmp_pwr[0][TENSORP] << std::endl;
    // lyhong_file << "TEXP: " << sample_Per_cmp_pwr[0][TEXP] << std::endl;
    // lyhong_file << "SCHEDP: " << sample_Per_cmp_pwr[0][SCHEDP] << std::endl;
    // lyhong_file << "IDLE_COREP: " << sample_Per_cmp_pwr[0][IDLE_COREP] << std::endl;
    // lyhong_file << "PIPEP: " << sample_Per_cmp_pwr[0][PIPEP] << std::endl;
    lyhong_file.flush();
    
    powerfile << "L2CP: " << sample_cmp_pwr[L2CP] << std::endl;
    powerfile << "MCP: " << sample_cmp_pwr[MCP] << std::endl;
    powerfile << "NOCP: " << sample_cmp_pwr[NOCP] << std::endl;
    powerfile << "DRAMP: " << sample_cmp_pwr[DRAMP] << std::endl;
    powerfile << "CONSTP: " << sample_cmp_pwr[CONSTP] << std::endl;
    powerfile << "STATICP: " << sample_cmp_pwr[STATICP] << std::endl;
    powerfile << "Sum: " << sample_cmp_pwr[L2CP] + sample_cmp_pwr[MCP] + sample_cmp_pwr[NOCP] + sample_cmp_pwr[DRAMP] + sample_cmp_pwr[CONSTP] + sample_cmp_pwr[STATICP] << std::endl;
    powerfile.flush();

    if(Percore_detail) {
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][INT_MULP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][INT_MUL24P] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][INT_MUL32P] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][INT_DIVP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][FP_DIVP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][FP_SQRTP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][FP_LGP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][FP_SINP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][DP_DIVP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][TENSORP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][TEXP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][DPUP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][DP_MULP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][FP_MULP] << "\t";
      for (unsigned i = 0; i < num_cores; i++)
        lyhong_SM_file << sample_Per_cmp_pwr[i][FP_EXP] << "\t";
      for (unsigned i = 0; i < num_cores; i++) {
        lyhong_SM_file << sample_Per_cmp_pwr[i][FPUP] << (i + 1 < num_cores ? "\t" : "");
      }
      lyhong_SM_file << std::endl;
      lyhong_SM_file.flush();
    }
  }
}

void gpgpu_sim_wrapper::compute() { proc->compute(); }
void gpgpu_sim_wrapper::print_power_kernel_stats(
    double gpu_sim_cycle, double gpu_tot_sim_cycle, double init_value,
    const std::string& kernel_info_string, bool print_trace) {
  detect_print_steady_state(1, init_value);
  if (g_power_simulation_enabled) {
    if(!Lyhong_Percore_sim) {
      powerfile << kernel_info_string << std::endl;

      sanity_check((kernel_power.avg * kernel_sample_count), kernel_tot_power);
      powerfile << "Kernel Average Power Data:" << std::endl;
      powerfile << "kernel_avg_power = " << kernel_power.avg << std::endl;

      for (unsigned i = 0; i < num_pwr_cmps; ++i) {
        powerfile << "gpu_avg_" << pwr_cmp_label[i] << " = "
                  << kernel_cmp_pwr[i].avg / kernel_sample_count << std::endl;
      }
      for (unsigned i = 0; i < num_perf_counters; ++i) {
        powerfile << "gpu_avg_" << perf_count_label[i] << " = "
                  << kernel_cmp_perf_counters[i].avg / kernel_sample_count
                  << std::endl;
      }

      powerfile << "gpu_avg_threads_per_warp = "
                << avg_threads_per_warp_tot / (double)kernel_sample_count
                << std::endl;

      for (unsigned i = 0; i < num_perf_counters; ++i) {
        powerfile << "gpu_tot_" << perf_count_label[i] << " = "
                  << kernel_cmp_perf_counters[i].avg << std::endl;
      }

      powerfile << std::endl << "Kernel Maximum Power Data:" << std::endl;
      powerfile << "kernel_max_power = " << kernel_power.max << std::endl;
      for (unsigned i = 0; i < num_pwr_cmps; ++i) {
        powerfile << "gpu_max_" << pwr_cmp_label[i] << " = "
                  << kernel_cmp_pwr[i].max << std::endl;
      }
      for (unsigned i = 0; i < num_perf_counters; ++i) {
        powerfile << "gpu_max_" << perf_count_label[i] << " = "
                  << kernel_cmp_perf_counters[i].max << std::endl;
      }

      powerfile << std::endl << "Kernel Minimum Power Data:" << std::endl;
      powerfile << "kernel_min_power = " << kernel_power.min << std::endl;
      for (unsigned i = 0; i < num_pwr_cmps; ++i) {
        powerfile << "gpu_min_" << pwr_cmp_label[i] << " = "
                  << kernel_cmp_pwr[i].min << std::endl;
      }
      for (unsigned i = 0; i < num_perf_counters; ++i) {
        powerfile << "gpu_min_" << perf_count_label[i] << " = "
                  << kernel_cmp_perf_counters[i].min << std::endl;
      }

      powerfile << std::endl
                << "Accumulative Power Statistics Over Previous Kernels:"
                << std::endl;
      powerfile << "gpu_tot_avg_power = "
                << gpu_tot_power.avg / total_sample_count << std::endl;
      powerfile << "gpu_tot_max_power = " << gpu_tot_power.max << std::endl;
      powerfile << "gpu_tot_min_power = " << gpu_tot_power.min << std::endl;
      powerfile << std::endl << std::endl;
      powerfile.flush();

      if (print_trace) {
        print_trace_files();
      }
    
      double sample_power = proc->rt_power.readOp.dynamic + sample_cmp_pwr[CONSTP] + sample_cmp_pwr[STATICP];
      lyhong_file << "Total_sample_count(TSC): " << total_sample_count << "      (time = TSC * gpgpu_runtime_stat / freq)"   << std::endl;
      lyhong_file << "Kernel_sample_count: " << kernel_sample_count<<std::endl;
      lyhong_file << "rt_power.readOp.dynamic: " << proc->rt_power.readOp.dynamic << "      (IDLE_COREP + gpu_*P)"  << std::endl;
      lyhong_file << "Total Power: " << sample_power << std::endl;
      for (unsigned i = 0; i < num_pwr_cmps; ++i) {
        lyhong_file << "gpu_" << pwr_cmp_label[i] << ": " << sample_cmp_pwr[i] << " " << std::endl;
      }
    } 
  }
}
void gpgpu_sim_wrapper::dump() {
  if (g_power_per_cycle_dump) proc->displayEnergy(2, 5);
}

void gpgpu_sim_wrapper::print_steady_state(int position, double init_val) {
  double temp_avg = sample_val / (double)samples.size();
  double temp_ipc = (init_val - init_inst_val) /
                    (double)(samples.size() * gpu_stat_sample_freq);

  if ((samples.size() >
       gpu_steady_min_period)) {  // If steady state occurred for some time,
                                  // print to file
    has_written_avg = true;
    gzprintf(steady_state_tacking_file, "%u,%d,%f,%f,", sample_start,
             total_sample_count, temp_avg, temp_ipc);
    for (unsigned i = 0; i < num_perf_counters; ++i) {
      gzprintf(steady_state_tacking_file, "%f,",
               samples_counter.at(i) / ((double)samples.size()));
    }
    gzprintf(steady_state_tacking_file, "\n");
  } else {
    if (!has_written_avg && position)
      gzprintf(steady_state_tacking_file,
               "ERROR! Not enough steady state points to generate average\n");
  }

  sample_start = 0;
  sample_val = 0;
  init_inst_val = init_val;
  samples.clear();
  samples_counter.clear();
  pwr_counter.clear();
  assert(samples.size() == 0);
}

void gpgpu_sim_wrapper::detect_print_steady_state(int position,
                                                  double init_val) {
  // Calculating Average
  if (g_power_simulation_enabled && g_steady_power_levels_enabled) {
    steady_state_tacking_file = gzopen(g_steady_state_tracking_filename, "a");
    if (position == 0) {
      if (samples.size() == 0) {
        // First sample
        sample_start = total_sample_count;
        sample_val = proc->rt_power.readOp.dynamic;
        init_inst_val = init_val;
        samples.push_back(proc->rt_power.readOp.dynamic);
        assert(samples_counter.size() == 0);
        assert(pwr_counter.size() == 0);

        for (unsigned i = 0; i < (num_perf_counters); ++i) {
          samples_counter.push_back(sample_perf_counters[i]);
        }

        for (unsigned i = 0; i < (num_pwr_cmps); ++i) {
          pwr_counter.push_back(sample_cmp_pwr[i]);
        }
        assert(pwr_counter.size() == (double)num_pwr_cmps);
        assert(samples_counter.size() == (double)num_perf_counters);
      } else {
        // Get current average
        double temp_avg = sample_val / (double)samples.size();

        if (abs(proc->rt_power.readOp.dynamic - temp_avg) <
            gpu_steady_power_deviation) {  // Value is within threshold
          sample_val += proc->rt_power.readOp.dynamic;
          samples.push_back(proc->rt_power.readOp.dynamic);
          for (unsigned i = 0; i < (num_perf_counters); ++i) {
            samples_counter.at(i) += sample_perf_counters[i];
          }

          for (unsigned i = 0; i < (num_pwr_cmps); ++i) {
            pwr_counter.at(i) += sample_cmp_pwr[i];
          }

        } else {  // Value exceeds threshold, not considered steady state
          print_steady_state(position, init_val);
        }
      }
    } else {
      print_steady_state(position, init_val);
    }
    gzclose(steady_state_tacking_file);
  }
}

void gpgpu_sim_wrapper::open_files() {
  if (g_power_simulation_enabled) {
    if (g_power_trace_enabled) {
      power_trace_file = gzopen(g_power_trace_filename, "a");
      metric_trace_file = gzopen(g_metric_trace_filename, "a");
    }
  }
}
void gpgpu_sim_wrapper::close_files() {
  if (g_power_simulation_enabled) {
    if (g_power_trace_enabled) {
      gzclose(power_trace_file);
      gzclose(metric_trace_file);
    }
  }
}
