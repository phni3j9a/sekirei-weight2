//! SOURCE ONLY dedicated positions update. The legacy method is untouched.
//! Compiled as child of trainer.rs, alongside frozen paired_nonlinear_adapter.
//! No file/process I/O or native/checkpoint serializer exists in this module.
use super::*;
use super::paired_nonlinear_adapter as paired;

pub const PROTOTYPE_ONLY: bool = true;
pub fn runtime_guard() -> Result<(), String> {
    Err("SOURCE ONLY train-position adapter disabled before I/O".into())
}

// FINITE OBSERVATION: no fused multiply-add, reassociation or replacement loss.
// Each check returns the original computed value when finite. On failure the
// existing outer Context::fail poisons the position/epoch before any save.
#[inline]
fn checked_finite(value:f32, stage:&'static str)->Result<f32,String> {
    if value.is_finite() {Ok(value)} else {Err(format!("nonfinite paired intermediate: {stage}"))}
}
#[inline]
fn checked_finite_f64(value:f64, stage:&'static str)->Result<f64,String> {
    if value.is_finite() {Ok(value)} else {Err(format!("nonfinite paired observation: {stage}"))}
}
#[inline]
fn checked_add(left:f32, right:f32, stage:&'static str)->Result<f32,String> {
    let sum=left+right;
    checked_finite(sum,stage)
}
#[inline]
fn checked_product(left:f32, right:f32, stage:&'static str)->Result<f32,String> {
    let product=left*right;
    checked_finite(product,stage)
}
#[inline]
fn checked_product_sum(acc:f32, left:f32, right:f32, stage:&'static str)->Result<f32,String> {
    // Validate the raw multiplication before addition can cancel or mask it.
    let product=checked_product(left,right,stage)?;
    checked_add(acc,product,stage)
}

/// Declaration-only context. Actual source/recipe/reference provenance must
/// be established by the future reader BEFORE this memory constructor.
pub struct Context {
    reference: TrainWeights,
    recipe: paired::DeclaredRecipe,
    poisoned: Option<String>,
    epochs_completed: u32,
}
impl Context {
    pub fn from_verified_declaration(reference: TrainWeights,
                                     recipe: paired::DeclaredRecipe) -> Result<Self, String> {
        paired::validate_recipe(recipe).map_err(str::to_string)?;
        Ok(Self { reference, recipe, poisoned:None, epochs_completed:0 })
    }
    pub fn ensure_live(&self) -> Result<(), String> {
        match &self.poisoned {
            Some(error) => Err(format!("terminal partial update; save/resume forbidden: {error}")),
            None => Ok(()),
        }
    }
    pub fn into_completed_reference(self)->Result<TrainWeights,String> {
        self.ensure_live()?;
        if self.epochs_completed!=3 {return Err("all three full epochs required before export".into());}
        Ok(self.reference)
    }
    fn fail<T>(&mut self, error: String) -> Result<T, String> {
        if self.poisoned.is_none() { self.poisoned=Some(error.clone()); }
        Err(error)
    }
}

/// Private fields prevent manufacturing a successful epoch commit token.
/// This is not a provenance, native-save, lifecycle or model-quality receipt.
pub struct EpochToken { epoch:u32, positions:usize, end_step:u64 }
impl EpochToken {
    pub fn epoch(&self)->u32 {self.epoch}
    pub fn positions(&self)->usize {self.positions}
    pub fn end_step(&self)->u64 {self.end_step}
}

impl Trainer {
    pub fn paired_optimizer_step(&self)->u64 {self.weights.step}
    fn paired_config_guard(&self) -> Result<(), String> {
        if self.residual_material_target || self.search_target_weight!=1.0
            || self.teacher_score_cap!=30000.0 || !self.exclude_mate_labels
            || self.teacher_time_limit.is_some() || self.teacher_node_limit.is_some()
            || self.grad_clip_norm.is_some() || self.ft_clip_norm.is_some()
            || self.l2_clip_norm.is_some() || self.out_clip_norm.is_some()
            || self.diagnostic_freeze_layer.is_some()
            || self.diagnostic_replay_component.is_some()
            || !self.diagnostic_shadow_trace_probe_boards.is_empty()
            || self.diagnostic_conflict_mask.is_some()
            || self.diagnostic_rate_matched_mask_count!=0
            || self.cp_wdl_grad_trace || self.sample_grad_trace_limit!=0
            || !self.trace_positions.is_empty() || self.weight_snapshot_trace {
            return Err("dedicated absolute CP-MSE mode rejects legacy alternate objectives/update/trace routes".into());
        }
        Ok(())
    }

    /// One original forward/backward, exactly one global step, one tied update.
    /// Any error poisons the context; no partially mutated state can continue.
    pub fn train_paired_nonlinear_position(&mut self, board:&Board, teacher_cp:i32,
                                         context:&mut Context)
        -> Result<paired::UpdateStats, String>
    {
        context.ensure_live()?;
        let result=(|| {
            self.paired_config_guard()?;
            if !self.lr.is_finite() || self.lr.to_bits()!=context.recipe.learning_rate.to_bits() {
                return Err("single declared learning rate/Trainer.lr bits mismatch".into());
            }
            if teacher_cp.abs_diff(0)>=30000 { return Err("integer absolute teacher outside allowed range".into()); }
            paired::validate_state(&self.weights,&context.reference,context.recipe).map_err(str::to_string)?;
            self.train_paired_nonlinear_position_inner(board,teacher_cp as f32,
                                                      &context.reference,context.recipe)
        })();
        match result { Ok(update)=>Ok(update), Err(error)=>context.fail(error) }
    }

    #[allow(clippy::too_many_arguments)]
    fn train_paired_nonlinear_position_inner(&mut self, board:&Board, teacher:f32,
        reference:&TrainWeights, recipe:paired::DeclaredRecipe)
        -> Result<paired::UpdateStats, String>
    {
        // These sentinels are identical to original positions mode.
        let weight=1.0_f32; let eval_teacher=teacher;
        let wdl_target:Option<f32>=None; let game_id=0_u64;
        let game_result=GameResult::Unknown;
        let stm = board.side_to_move;
        let w = &self.weights;

        // ── Forward pass ──────────────────────────────────────────────────────

        // FT accumulation
        let mut acc_us = w.ft_bias.clone();
        let mut acc_them = acc_us.clone();

        let active_us = active_features(board, stm);
        let active_them = active_features(board, stm.flip());

        for feat in &active_us {
            let base = feat * L1;
            for j in 0..L1 {
                acc_us[j] = checked_add(acc_us[j],w.ft[base+j],"forward.ft.us.add")?;
            }
        }
        for feat in &active_them {
            let base = feat * L1;
            for j in 0..L1 {
                acc_them[j] = checked_add(acc_them[j],w.ft[base+j],"forward.ft.them.add")?;
            }
        }

        // FT ClippedReLU [0, 127]
        let relu_us: Vec<f32> = acc_us.iter().map(|&x| x.clamp(0.0, 127.0)).collect();
        let relu_them: Vec<f32> = acc_them.iter().map(|&x| x.clamp(0.0, 127.0)).collect();
        for &x in relu_us.iter().chain(relu_them.iter()) {
            self.ft_output_sum += x as f64;
            self.ft_output_sum_sq += (x as f64) * (x as f64);
        }
        self.ft_output_count += 2 * L1 as u64;
        for j in 0..L1 {
            if relu_us[j] > 0.0 || relu_them[j] > 0.0 {
                self.ft_ever_active[j] = true;
            }
            if relu_us[j] >= 127.0 || relu_them[j] >= 127.0 {
                self.ft_ever_saturated[j] = true;
            }
            // Frequency-based counterparts of the ever-flags above (see
            // `Trainer::ft_zero_count`'s doc comment): "dead" is the
            // logical complement of `ft_ever_active`'s OR (neither
            // perspective fires this position), "saturated" mirrors
            // `ft_ever_saturated`'s OR directly (either perspective
            // saturates).
            if acc_us[j] <= 0.0 && acc_them[j] <= 0.0 {
                self.ft_zero_count[j] += 1;
            }
            if acc_us[j] >= 127.0 || acc_them[j] >= 127.0 {
                self.ft_sat_count[j] += 1;
            }
        }

        // L2 accumulation
        let mut l2_acc = w.l2_bias.clone(); // Vec<f32> len=L2
        for j in 0..L1 {
            let a = relu_us[j];
            let b = relu_them[j];
            let base_us = j * L2;
            let base_them = (L1 + j) * L2;
            for o in 0..L2 {
                l2_acc[o] = checked_product_sum(l2_acc[o],a,w.l2[base_us+o],"forward.l2.us.mul-add")?;
                l2_acc[o] = checked_product_sum(l2_acc[o],b,w.l2[base_them+o],"forward.l2.them.mul-add")?;
            }
        }

        // L2 ClippedReLU [0, 127]
        let relu_l2: Vec<f32> = l2_acc.iter().map(|&x| x.clamp(0.0, 127.0)).collect();
        for o in 0..L2 {
            if relu_l2[o] > 0.0 {
                self.l2_ever_active[o] = true;
            }
            if relu_l2[o] >= 127.0 {
                self.l2_ever_saturated[o] = true;
            }
            let pre = l2_acc[o];
            if pre <= 0.0 {
                self.l2_zero_count[o] += 1;
            }
            if pre >= 127.0 {
                self.l2_sat_count[o] += 1;
            }
            self.l2_values[o].push(pre);
            // `pre` is `weighted_input + l2_bias[o]` (see the accumulation
            // loop above, which starts from `w.l2_bias.clone()`) -- the
            // weighted-input term alone, for `--trace-positions`'s
            // bias-vs-weight-input split.
            self.l2_weighted_input_values[o].push(checked_finite(pre-w.l2_bias[o],"observation.l2.weighted-input")?);
        }
        self.l2_sample_count += 1;
        let l2_input_norm_sq: f64 = relu_us
            .iter()
            .chain(relu_them.iter())
            .map(|&x| (x as f64).powi(2))
            .sum();
        self.l2_input_norm_sum += l2_input_norm_sq.sqrt();
        self.l2_input_norm_sq_sum += l2_input_norm_sq;

        // Output
        let mut output = w.out_bias;
        for o in 0..L2 {
            output = checked_product_sum(output,relu_l2[o],w.out[o],"forward.output.mul-add")?;
        }
        let score = checked_finite(output/64.0,"forward.score")?;
        self.output_sum += score as f64;
        self.output_sum_sq += (score as f64) * (score as f64);

        // ── Loss ──────────────────────────────────────────────────────────────

        let err = checked_finite(score-teacher,"loss.error")?;
        let cp_mse = checked_product(err,err,"loss.cp-mse")?;
        self.total_loss += (weight as f64) * cp_mse as f64;
        checked_finite_f64(self.total_loss,"observation.total-loss")?;
        self.total_count += 1;
        self.total_weight += weight as f64;

        // Diagnostic-only: target/prediction distribution and their
        // relationship, and the loss split into its CP/WDL components --
        // none of this feeds the gradient below, which is still computed
        // from `err` (score - blended teacher) exactly as before.
        self.target_sum += teacher as f64;
        self.target_sum_sq += (teacher as f64) * (teacher as f64);
        self.eval_teacher_sum += eval_teacher as f64;
        self.eval_teacher_sum_sq += (eval_teacher as f64) * (eval_teacher as f64);
        self.pred_eval_prod_sum += (score as f64) * (eval_teacher as f64);
        let cp_err = (score - eval_teacher) as f64;
        self.cp_component_sum += cp_err * cp_err;
        if let Some(wdl_target) = wdl_target {
            let wdl_err = (score - wdl_target) as f64;
            self.wdl_component_sum += wdl_err * wdl_err;
            self.wdl_component_count += 1;

            // `--cp-wdl-grad-trace`: two extra, diagnostic-only backward
            // passes -- one with `eval_teacher` as the sole teacher, one
            // with `wdl_target` -- decomposing the blended gradient below
            // into its two contributions. Never applied to `self.weights`;
            // see `diagnostic_backward`'s doc comment.
            if self.cp_wdl_grad_trace {
                let cp = diagnostic_backward(
                    &self.weights,
                    &l2_acc,
                    &relu_l2,
                    &acc_us,
                    &acc_them,
                    &relu_us,
                    &relu_them,
                    &active_us,
                    &active_them,
                    score - eval_teacher,
                    weight,
                );
                let wdl = diagnostic_backward(
                    &self.weights,
                    &l2_acc,
                    &relu_l2,
                    &acc_us,
                    &acc_them,
                    &relu_us,
                    &relu_them,
                    &active_us,
                    &active_them,
                    score - wdl_target,
                    weight,
                );
                for o in 0..L2 {
                    let gc = cp.d_l2_acc[o] as f64;
                    let gw = wdl.d_l2_acc[o] as f64;
                    self.l2_cp_dacc_sum[o] += gc;
                    self.l2_cp_dacc_sq_sum[o] += gc * gc;
                    self.l2_wdl_dacc_sum[o] += gw;
                    self.l2_wdl_dacc_sq_sum[o] += gw * gw;
                    self.l2_cp_wdl_dot_sum[o] += gc * gw;
                    if gc > 0.0 {
                        self.l2_cp_dacc_pos_count[o] += 1;
                    } else if gc < 0.0 {
                        self.l2_cp_dacc_neg_count[o] += 1;
                    }
                    if gw > 0.0 {
                        self.l2_wdl_dacc_pos_count[o] += 1;
                    } else if gw < 0.0 {
                        self.l2_wdl_dacc_neg_count[o] += 1;
                    }
                }
                for j in 0..L1 {
                    let gc = cp.d_ft_acc[j] as f64;
                    let gw = wdl.d_ft_acc[j] as f64;
                    self.ft_cp_dacc_sum[j] += gc;
                    self.ft_cp_dacc_sq_sum[j] += gc * gc;
                    self.ft_wdl_dacc_sum[j] += gw;
                    self.ft_wdl_dacc_sq_sum[j] += gw * gw;
                    self.ft_cp_wdl_dot_sum[j] += gc * gw;
                    if gc > 0.0 {
                        self.ft_cp_dacc_pos_count[j] += 1;
                    } else if gc < 0.0 {
                        self.ft_cp_dacc_neg_count[j] += 1;
                    }
                    if gw > 0.0 {
                        self.ft_wdl_dacc_pos_count[j] += 1;
                    } else if gw < 0.0 {
                        self.ft_wdl_dacc_neg_count[j] += 1;
                    }
                }
                self.cp_ft_grad_norm_sum += cp.ft_grad_norm;
                self.cp_ft_grad_norm_sum_sq += cp.ft_grad_norm * cp.ft_grad_norm;
                self.wdl_ft_grad_norm_sum += wdl.ft_grad_norm;
                self.wdl_ft_grad_norm_sum_sq += wdl.ft_grad_norm * wdl.ft_grad_norm;
                self.cp_l2_grad_norm_sum += cp.l2_grad_norm;
                self.cp_l2_grad_norm_sum_sq += cp.l2_grad_norm * cp.l2_grad_norm;
                self.wdl_l2_grad_norm_sum += wdl.l2_grad_norm;
                self.wdl_l2_grad_norm_sum_sq += wdl.l2_grad_norm * wdl.l2_grad_norm;
                self.cp_out_grad_norm_sum += cp.out_grad_norm;
                self.cp_out_grad_norm_sum_sq += cp.out_grad_norm * cp.out_grad_norm;
                self.wdl_out_grad_norm_sum += wdl.out_grad_norm;
                self.wdl_out_grad_norm_sum_sq += wdl.out_grad_norm * wdl.out_grad_norm;

                // Target/prediction/residual/dL-dOutput distributions --
                // explains *why* the gradient-scale fields above differ,
                // not just that they do. Scoped to this same wdl-having
                // subset (not the epoch-wide `eval_teacher_sum`/
                // `output_sum`), so every field is a fair comparison over
                // the identical position set.
                let eval_teacher_f64 = eval_teacher as f64;
                let wdl_target_f64 = wdl_target as f64;
                let score_f64 = score as f64;
                self.cp_target_sum += eval_teacher_f64;
                self.cp_target_sum_sq += eval_teacher_f64 * eval_teacher_f64;
                self.wdl_target_sum += wdl_target_f64;
                self.wdl_target_sum_sq += wdl_target_f64 * wdl_target_f64;
                self.prediction_sum += score_f64;
                self.prediction_sum_sq += score_f64 * score_f64;
                self.cp_residual_sum += cp_err;
                self.cp_residual_sum_sq += cp_err * cp_err;
                self.wdl_residual_sum += wdl_err;
                self.wdl_residual_sum_sq += wdl_err * wdl_err;
                let cp_d_output = cp.d_output as f64;
                let wdl_d_output = wdl.d_output as f64;
                self.cp_d_output_sum += cp_d_output;
                self.cp_d_output_sum_sq += cp_d_output * cp_d_output;
                self.wdl_d_output_sum += wdl_d_output;
                self.wdl_d_output_sum_sq += wdl_d_output * wdl_d_output;
            }
        }

        // ── Backward pass ─────────────────────────────────────────────────────

        let d_score = checked_product(checked_product(weight,2.0,"backward.d-score.weight")?,err,"backward.d-score.error")?;
        let d_output = checked_finite(d_score/64.0,"backward.d-output")?;

        // Output layer gradients
        let mut d_out = vec![0.0f32; L2];
        for o in 0..L2 {
            d_out[o] = checked_product(d_output,relu_l2[o],"backward.output.gradient")?;
        }
        let mut d_out_bias = d_output;

        // Backprop through L2 ClippedReLU
        let mut d_l2_acc = [0.0f32; L2];
        for o in 0..L2 {
            // Observe the raw product BEFORE an inactive ClippedReLU hides it.
            let raw = checked_product(d_output,self.weights.out[o],"backward.l2.raw-before-mask")?;
            if l2_acc[o] > 0.0 && l2_acc[o] < 127.0 {
                d_l2_acc[o] = raw;
            }
        }
        // `d_l2_acc[o]` is the gradient of the loss w.r.t. neuron o's own
        // pre-activation -- the per-neuron trace's most direct "which wall
        // is this neuron being pushed toward" signal (see
        // `Trainer::l2_dacc_sum`'s doc comment).
        for o in 0..L2 {
            let g = d_l2_acc[o] as f64;
            self.l2_dacc_sum[o] += g;
            self.l2_dacc_sq_sum[o] += g * g;
            if g > 0.0 {
                self.l2_dacc_pos_count[o] += 1;
            } else if g < 0.0 {
                self.l2_dacc_neg_count[o] += 1;
            }
        }

        // `--sample-grad-trace`: one record per position, up to the
        // requested limit, from the real blended `d_l2_acc` above -- never
        // reorders or otherwise changes what training does, purely reads
        // already-computed forward/backward state (see
        // `Trainer::sample_grad_trace_limit`'s doc comment).
        if self.sample_grad_trace_limit > 0 && self.l2_sample_count <= self.sample_grad_trace_limit
        {
            let cp_d_output = weight * 2.0 * (score - eval_teacher) / 64.0;
            let wdl_d_output = wdl_target.map(|t| weight * 2.0 * (score - t) / 64.0);
            let l2_grad_norm = (d_l2_acc.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()).sqrt();
            let cosine_prev = self
                .sample_grad_prev_d_l2_acc
                .as_ref()
                .map(|prev| diagnostics::vector_cosine_similarity(prev, &d_l2_acc));
            let cosine_running_mean = if self.sample_grad_running_count > 0 {
                Some(diagnostics::vector_cosine_similarity(
                    &self.sample_grad_running_mean_d_l2_acc,
                    &d_l2_acc,
                ))
            } else {
                None
            };
            let l2_gate: Vec<i8> = l2_acc
                .iter()
                .map(|&x| {
                    if x <= 0.0 {
                        -1
                    } else if x >= 127.0 {
                        1
                    } else {
                        0
                    }
                })
                .collect();
            self.sample_grad_records
                .push(diagnostics::SampleGradRecord {
                    game_id,
                    game_result: format!("{game_result:?}"),
                    position_index: self.l2_sample_count,
                    prediction: score,
                    cp_target: eval_teacher,
                    wdl_target,
                    cp_d_output,
                    wdl_d_output,
                    l2_grad_vector: d_l2_acc.to_vec(),
                    l2_grad_norm,
                    cosine_prev,
                    cosine_running_mean,
                    l2_gate,
                });
            self.sample_grad_prev_d_l2_acc = Some(d_l2_acc);
            self.sample_grad_running_count += 1;
            let n = self.sample_grad_running_count as f32;
            for o in 0..L2 {
                self.sample_grad_running_mean_d_l2_acc[o] +=
                    (d_l2_acc[o] - self.sample_grad_running_mean_d_l2_acc[o]) / n;
            }
        }

        // L2 weight gradients and propagate to FT
        let mut d_l2 = vec![0.0f32; 2 * L1 * L2];
        let mut d_l2_bias = vec![0.0f32; L2];
        let mut d_relu_us = vec![0.0f32; L1];
        let mut d_relu_them = vec![0.0f32; L1];

        for j in 0..L1 {
            let base_us = j * L2;
            let base_them = (L1 + j) * L2;
            for o in 0..L2 {
                let g = d_l2_acc[o];
                d_l2[base_us+o] = checked_product_sum(d_l2[base_us+o],g,relu_us[j],"backward.l2.us-gradient.mul-add")?;
                d_l2[base_them+o] = checked_product_sum(d_l2[base_them+o],g,relu_them[j],"backward.l2.them-gradient.mul-add")?;
                d_relu_us[j] = checked_product_sum(d_relu_us[j],g,self.weights.l2[base_us+o],"backward.ft.us-raw.mul-add")?;
                d_relu_them[j] = checked_product_sum(d_relu_them[j],g,self.weights.l2[base_them+o],"backward.ft.them-raw.mul-add")?;
            }
        }
        d_l2_bias[..L2].copy_from_slice(&d_l2_acc[..L2]);

        // Backprop through FT ClippedReLU
        let mut d_acc_us = vec![0.0f32; L1];
        let mut d_acc_them = vec![0.0f32; L1];
        for j in 0..L1 {
            checked_finite(d_relu_us[j],"backward.ft.us-raw-before-mask")?;
            checked_finite(d_relu_them[j],"backward.ft.them-raw-before-mask")?;
            if acc_us[j] > 0.0 && acc_us[j] < 127.0 {
                d_acc_us[j] = d_relu_us[j];
            }
            if acc_them[j] > 0.0 && acc_them[j] < 127.0 {
                d_acc_them[j] = d_relu_them[j];
            }
        }

        // FT weight gradients (sparse)
        let mut d_ft = vec![0.0f32; INPUT * L1];
        let mut d_bias = vec![0.0f32; L1];

        for feat in &active_us {
            let base = feat * L1;
            for j in 0..L1 {
                d_ft[base+j] = checked_add(d_ft[base+j],d_acc_us[j],"backward.ft.us-sparse.add")?;
            }
        }
        for feat in &active_them {
            let base = feat * L1;
            for j in 0..L1 {
                d_ft[base+j] = checked_add(d_ft[base+j],d_acc_them[j],"backward.ft.them-sparse.add")?;
            }
        }
        for j in 0..L1 {
            d_bias[j] = checked_add(d_acc_us[j],d_acc_them[j],"backward.ft.bias-gradient.add")?;
        }
        // `d_bias[j]` (the FT bias gradient) is exactly the gradient of the
        // loss w.r.t. neuron j's own pre-activation, summed across both
        // perspectives -- FT's direct counterpart to `d_l2_acc` above.
        for j in 0..L1 {
            let g = d_bias[j] as f64;
            self.ft_dacc_sum[j] += g;
            self.ft_dacc_sq_sum[j] += g * g;
            if g > 0.0 {
                self.ft_dacc_pos_count[j] += 1;
            } else if g < 0.0 {
                self.ft_dacc_neg_count[j] += 1;
            }
        }

        // ── Gradient-norm diagnostics ────────────────────────────────────────
        // Diagnostic-only, computed from the gradients above without altering
        // them. `d_ft`'s only nonzero entries are the rows touched by
        // `active_us`/`active_them`, and `d_ft[base+j] == d_acc_us[j]` (or
        // `d_acc_them[j]`) for every touched row of that side -- so its
        // squared-norm contribution is exactly `active_us.len() * Σ
        // d_acc_us[j]²` plus the `active_them` term, without a second pass
        // over the full `INPUT*L1`-length array.
        //
        // ponytail: this slightly over-counts in the (architecture-rare)
        // case where the same feature index appears in both `active_us` and
        // `active_them`, since that row's true `d_ft` value is their sum,
        // not two independent entries -- acceptable for a monitoring metric.
        let d_acc_us_sq: f64 = d_acc_us.iter().map(|&x| (x as f64).powi(2)).sum();
        let d_acc_them_sq: f64 = d_acc_them.iter().map(|&x| (x as f64).powi(2)).sum();
        let d_bias_sq: f64 = d_bias.iter().map(|&x| (x as f64).powi(2)).sum();
        let ft_grad_sq = d_acc_us_sq * active_us.len() as f64
            + d_acc_them_sq * active_them.len() as f64
            + d_bias_sq;
        let l2_grad_sq: f64 = d_l2.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()
            + d_l2_bias.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
        let out_grad_sq: f64 =
            d_out.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() + (d_out_bias as f64).powi(2);

        let ft_grad_norm = checked_finite_f64(ft_grad_sq.sqrt(),"observation.ft-gradient-norm")?;
        let l2_grad_norm = checked_finite_f64(l2_grad_sq.sqrt(),"observation.l2-gradient-norm")?;
        let out_grad_norm = checked_finite_f64(out_grad_sq.sqrt(),"observation.output-gradient-norm")?;
        self.ft_grad_norm_sum += ft_grad_norm;
        self.ft_grad_norm_sum_sq += ft_grad_norm * ft_grad_norm;
        self.l2_grad_norm_sum += l2_grad_norm;
        self.l2_grad_norm_sum_sq += l2_grad_norm * l2_grad_norm;
        self.out_grad_norm_sum += out_grad_norm;
        self.out_grad_norm_sum_sq += out_grad_norm * out_grad_norm;
        self.out_grad_norm_values.push(checked_finite(out_grad_norm as f32,"observation.output-gradient-norm-f32")?);
        let global_grad_norm = checked_finite_f64((ft_grad_sq+l2_grad_sq+out_grad_sq).sqrt(),"observation.global-gradient-norm")?;
        self.global_grad_norm_values.push(checked_finite(global_grad_norm as f32,"observation.global-gradient-norm-f32")?);

        // ── Per-layer gradient clipping (optional) ───────────────────────────
        // Each layer's gradient is compared against *its own* norm and *its
        // own* threshold, independent of the other layers -- unlike the
        // global-norm clipping below, setting only `out_clip_norm` leaves
        // FT/L2 completely untouched (a real single-variable change). Applied
        // before the diagnostics-vs-clip ordering matters the same way as
        // global clipping: the sums/percentiles above already captured the
        // unclipped norms, so this can't retroactively change what a
        // threshold-selection read from this run's own output.
        if let Some(clip_norm) = self.ft_clip_norm {
            let clip_norm = clip_norm as f64;
            if ft_grad_norm > clip_norm {
                self.ft_clip_count += 1;
                let scale = (clip_norm / ft_grad_norm) as f32;
                d_ft.iter_mut().for_each(|x| *x *= scale);
                d_bias.iter_mut().for_each(|x| *x *= scale);
            }
        }
        if let Some(clip_norm) = self.l2_clip_norm {
            let clip_norm = clip_norm as f64;
            if l2_grad_norm > clip_norm {
                self.l2_clip_count += 1;
                let scale = (clip_norm / l2_grad_norm) as f32;
                d_l2.iter_mut().for_each(|x| *x *= scale);
                d_l2_bias.iter_mut().for_each(|x| *x *= scale);
            }
        }
        let mut out_grad_norm_after = out_grad_norm;
        if let Some(clip_norm) = self.out_clip_norm {
            let clip_norm = clip_norm as f64;
            if out_grad_norm > clip_norm {
                self.out_clip_count += 1;
                let scale = (clip_norm / out_grad_norm) as f32;
                d_out.iter_mut().for_each(|x| *x *= scale);
                d_out_bias *= scale;
                out_grad_norm_after = clip_norm;
            }
        }
        self.out_grad_norm_after_sum += out_grad_norm_after;
        self.out_grad_norm_after_sum_sq += out_grad_norm_after * out_grad_norm_after;

        // ── Global gradient clipping (optional) ──────────────────────────────
        // Global-norm clipping: if the whole-network gradient norm exceeds
        // `grad_clip_norm`, scale every layer's gradient down by the same
        // factor (direction preserved, only magnitude reduced). Applied
        // after the diagnostics above capture the unclipped norm, so
        // `global_grad_norm_p95`/`p99` always describe the natural
        // distribution a threshold should be chosen from, not a value
        // that's already been clamped by whatever threshold is active.
        // Independent of the per-layer clipping above -- if both are set,
        // this acts on whatever the per-layer step already produced (an
        // untested combination; the 2026-07 experiments use exactly one
        // clipping mechanism at a time).
        if let Some(clip_norm) = self.grad_clip_norm {
            let clip_norm = clip_norm as f64;
            if global_grad_norm > clip_norm {
                self.grad_clip_count += 1;
                let scale = (clip_norm / global_grad_norm) as f32;
                d_ft.iter_mut().for_each(|x| *x *= scale);
                d_bias.iter_mut().for_each(|x| *x *= scale);
                d_l2.iter_mut().for_each(|x| *x *= scale);
                d_l2_bias.iter_mut().for_each(|x| *x *= scale);
                d_out.iter_mut().for_each(|x| *x *= scale);
                d_out_bias *= scale;
            }
        }

        // `--diagnostic-conflict-mask`/`--diagnostic-rate-matched-mask-*`:
        // stop the targeted layer(s)' update for this position only, either
        // because the prediction sits between the two teachers (real
        // signal) or because a seeded, conflict-independent draw selected
        // this position (rate-matched control) -- see `ConflictMaskLayer`'s
        // and `Trainer::rate_matched_should_mask`'s doc comments. Tracked
        // per group (conflict / non-conflict) regardless of which
        // mechanism (if any) is actually active this run, so the analysis
        // can confirm the masked positions are the dangerous ones -- not
        // just wherever the RNG happened to land.
        let eligible = wdl_target.is_some();
        if eligible {
            let wdl_target = wdl_target.expect("eligible checked wdl_target.is_some()");
            let cp_residual = (score - eval_teacher) as f64;
            let wdl_residual = (score - wdl_target) as f64;
            let is_conflict = cp_residual * wdl_residual < 0.0;

            let (mask_ft, mask_l2) = match self.diagnostic_conflict_mask {
                Some(ConflictMaskLayer::Ft) => (is_conflict, false),
                Some(ConflictMaskLayer::FtAndL2) => (is_conflict, is_conflict),
                None if self.diagnostic_rate_matched_mask_count > 0 => {
                    (self.rate_matched_should_mask(), false)
                }
                None => (false, false),
            };

            let dead_before_ft = acc_us
                .iter()
                .chain(acc_them.iter())
                .filter(|&&x| x.clamp(0.0, 127.0) == 0.0)
                .count() as u64;
            let dead_before_l2 = l2_acc.iter().filter(|&&x| x <= 0.0).count() as u64;

            let group = if is_conflict {
                &mut self.conflict_group
            } else {
                &mut self.nonconflict_group
            };
            group.count += 1;
            group.cp_residual_abs_sum += cp_residual.abs();
            group.cp_residual_abs_sq_sum += cp_residual * cp_residual;
            group.wdl_residual_abs_sum += wdl_residual.abs();
            group.wdl_residual_abs_sq_sum += wdl_residual * wdl_residual;
            group.ft_grad_norm_sum += ft_grad_norm;
            group.ft_grad_norm_sq_sum += ft_grad_norm * ft_grad_norm;
            group.l2_grad_norm_sum += l2_grad_norm;
            group.l2_grad_norm_sq_sum += l2_grad_norm * l2_grad_norm;

            if mask_ft {
                d_ft.iter_mut().for_each(|x| *x = 0.0);
                d_bias.iter_mut().for_each(|x| *x = 0.0);
            }
            if mask_l2 {
                d_l2.iter_mut().for_each(|x| *x = 0.0);
                d_l2_bias.iter_mut().for_each(|x| *x = 0.0);
            }
            if mask_ft || mask_l2 {
                self.masked_position_count += 1;
            }
            self.pending_conflict_dead_before = Some((is_conflict, dead_before_ft, dead_before_l2));
        } else {
            self.pending_conflict_dead_before = None;
        }

        // ── Adam update ───────────────────────────────────────────────────────

        self.weights.step = self.weights.step.checked_add(1)
            .ok_or_else(|| "global optimizer step overflow".to_string())?;
        let t = self.weights.step;
        let lr = self.lr;

        // No original layer/scalar/shadow Adam call follows this point.
        // DenseGradients contains pre-update physical gradients; adapter owns
        // SUM/sign aggregation, all inactive moments and protected skips.
        let update=paired::apply_dense_update(
            &mut self.weights, reference,
            paired::DenseGradients { ft:&d_ft, l2:&d_l2,
                l2_bias:&d_l2_bias, out:&d_out }, recipe,
        ).map_err(str::to_string)?;
        // Do not populate legacy physical-delta update traces from master data.
        // Original diagnostic accumulation is observational; UpdateStats is
        // the distinct logical-master trace and is returned to the epoch.
        Ok(update)
    }

    /// Complete keyed original teacher cache only; never searches or mutates
    /// cache. The caller orders samples with original shuffled_order when the
    /// externally frozen recipe requests it. No resume cursor/chunk path.
    pub fn train_paired_nonlinear_epoch(&mut self,
        samples:&[crate::positions::PositionSample], cache:&HashMap<String,i32>,
        epoch:u32, context:&mut Context) -> Result<EpochToken,String>
    {
        context.ensure_live()?;
        let result=(|| {
            self.paired_config_guard()?;
            if samples.is_empty() || cache.len()!=samples.len()
                || epoch!=context.epochs_completed+1 || epoch>3 {
                return Err("complete fresh sequential epoch/cache required".into());
            }
            let start_step=self.weights.step;
            let expected_start=(epoch as u64-1).checked_mul(samples.len() as u64)
                .ok_or_else(|| "epoch count overflow".to_string())?;
            if start_step!=expected_start { return Err("fresh full epoch/global step mismatch; resume forbidden".into()); }
            let mut seen=std::collections::HashSet::new();
            for sample in samples {
                let sfen=sekirei_core::sfen::board_to_sfen(&sample.board);
                let cp=cache.get(&sfen).ok_or_else(|| "missing keyed teacher".to_string())?;
                if !seen.insert(sfen) || cp.abs_diff(0)>=30000 {
                    return Err("duplicate position or teacher outside absolute integer domain".into());
                }
            }
            self.reset_epoch_stats();
            for (position_index,sample) in samples.iter().enumerate() {
                let sfen=sekirei_core::sfen::board_to_sfen(&sample.board);
                self.train_paired_nonlinear_position(&sample.board,cache[&sfen],context)?;
                if (position_index+1)%1024==0 || position_index+1==samples.len() {
                    eprintln!("PAIRED_POSITION_PROGRESS epoch={epoch} positions={} global_step={}",position_index+1,self.weights.step);
                }
            }
            let expected_end=start_step.checked_add(samples.len() as u64)
                .ok_or_else(|| "epoch step overflow".to_string())?;
            if self.weights.step!=expected_end || self.total_count!=samples.len() as u64 {
                return Err("epoch did not process every original train position exactly once".into());
            }
            paired::validate_state(&self.weights,&context.reference,context.recipe).map_err(str::to_string)?;
            Ok(EpochToken { epoch, positions:samples.len(), end_step:expected_end })
        })();
        match result {
            Ok(token)=>{ context.epochs_completed=epoch; Ok(token) },
            Err(error)=>context.fail(error),
        }
    }
}

// SOURCE-only test definitions: Root must compile/run with the real parent.
#[cfg(test)]
mod finite_observation_tests {
    use super::*;
    #[test]
    fn product_and_sum_overflow_are_rejected_before_clamp() {
        assert!(checked_product(f32::MAX,2.0,"product").is_err());
        assert!(checked_add(f32::MAX,f32::MAX,"sum").is_err());
        assert!(checked_product_sum(-f32::MAX,f32::MAX,2.0,"raw-before-cancel").is_err());
    }
    #[test]
    fn nonfinite_raw_gradient_cannot_hide_behind_inactive_mask() {
        let inactive=false;
        let raw=checked_product(f32::MAX,2.0,"raw-before-mask");
        assert!(!inactive && raw.is_err());
        assert!(checked_finite(f32::NAN,"NaN").is_err());
        assert!(checked_finite(f32::INFINITY,"Inf").is_err());
        assert!(checked_finite_f64(f64::INFINITY,"observation").is_err());
    }
    #[test]
    fn finite_result_and_signed_zero_bits_are_preserved() {
        assert_eq!(checked_product(-0.0,1.0,"zero").unwrap().to_bits(),(-0.0_f32).to_bits());
        assert_eq!(checked_add(-0.0,-0.0,"zero").unwrap().to_bits(),(-0.0_f32).to_bits());
        assert_eq!(checked_product_sum(4.0,2.0,3.0,"finite").unwrap().to_bits(),10.0_f32.to_bits());
    }
    #[test]
    fn separate_product_add_and_runtime_barrier_are_retained() {
        let a=f32::from_bits(0x3f80_0001);let b=f32::from_bits(0x3f7f_ffff);
        let expected=(-1.0_f32)+(a*b);
        assert_eq!(checked_product_sum(-1.0,a,b,"separate").unwrap().to_bits(),expected.to_bits());
        assert!(PROTOTYPE_ONLY && runtime_guard().is_err());
    }
}

// ADD-ONLY SOURCE test module. Root appends this file to paired_nonlinear_positions.rs.
// Root alone compiles/runs with SEKIREI_TRAIN_FTZ_DAZ=1; no production API changes.
// One public fixture x three epochs is not the original 112681-position fit.
#[cfg(test)]
mod fullshape_synthetic_epoch_tests {
    use super::*;
    use crate::trainer::paired_nonlinear_checkpoint_io as checkpoint;
    use crate::trainer::paired_nonlinear_checkpoint_native as native;
    use crate::positions::PositionSample;
    use sekirei_core::sfen::board_to_sfen;
    use std::collections::HashMap;

    const PUBLIC_AFTER_7G7F: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2";

    fn selected_recipe() -> paired::DeclaredRecipe {
        paired::DeclaredRecipe {
            learning_rate: f32::from_bits(0x3a83126f),
            head_init_width: f32::from_bits(0x3b800000),
            head_bias_init: f32::from_bits(0x40800000),
            output_native_l1_budget: f32::from_bits(0x47000000),
        }
    }

    fn actual_float_policy() {
        // The test harness does not call main. Use the existing main setting
        // then independently read the calling thread's actual control word.
        // Do not set the environment or invent a second FP implementation.
        assert_eq!(std::env::var("SEKIREI_TRAIN_FTZ_DAZ").unwrap(), "1");
        crate::configure_training_floats();
        let snapshot = crate::paired_nonlinear_float::runtime_ready().unwrap();
        assert_eq!(snapshot.environment_value, "1");
        assert_eq!(snapshot.mxcsr_control_bits, 0x9fc0);
        assert_eq!(snapshot.mxcsr_raw_bits & !0x3f, 0x9fc0);
    }

    fn public_sample() -> PositionSample {
        PositionSample {
            board: Board::from_sfen(PUBLIC_AFTER_7G7F).unwrap(),
            phase: "opening".into(), side_to_move: "white".into(), ply: 2,
            source: "public-technical-fixture-after-7g7f".into(),
        }
    }

    fn ordinary_cache(sample: &PositionSample) -> HashMap<String, i32> {
        HashMap::from([(board_to_sfen(&sample.board), 500)])
    }

    fn memory_checkpoint_context() -> checkpoint::CheckpointContext {
        // Identity-shape fixture only. No file with this path is read or made;
        // it is not a provenance declaration for an actual fit.
        let fullref = serde_json::json!({"path":"/tmp/public-synthetic-epoch-context", "bytes":1,
            "sha256":"ab".repeat(32)});
        checkpoint::CheckpointContext {
            recipe:fullref.clone(), source_binding:fullref.clone(), manifest:fullref.clone(),
            reference03:fullref.clone(), positions:fullref.clone(), labels:fullref,
            source_files:serde_json::json!({"/tmp/public-synthetic-epoch-context":{
                "bytes":1,"sha256":"ab".repeat(32)}}),
        }
    }

    #[test]
    fn fullshape_synthetic_three_epochs_float_policy_native_and_no_formal_export() {
        actual_float_policy();
        let recipe=selected_recipe();
        let (mut trainer,reference)=checkpoint::fresh_trainer(recipe).unwrap();
        assert_eq!(trainer.paired_optimizer_step(),0);
        trainer.paired_config_guard().unwrap();
        native::validate_checkpoint_state(&trainer.weights,recipe,
            native::SnapshotStage::Initialized,native::FEATURE_TAG).unwrap();
        let mut context=Context::from_verified_declaration(reference,recipe).unwrap();
        let samples=[public_sample()];
        let cache=ordinary_cache(&samples[0]);
        for epoch in 1..=3_u32 {
            actual_float_policy();
            let token=trainer.train_paired_nonlinear_epoch(&samples,&cache,epoch,&mut context).unwrap();
            assert_eq!(token.epoch(),epoch);
            assert_eq!(token.positions(),1);
            assert_eq!(token.end_step(),u64::from(epoch));
            assert_eq!(trainer.paired_optimizer_step(),u64::from(epoch));
            native::validate_checkpoint_state(&trainer.weights,recipe,
                native::SnapshotStage::AfterPosition{expected_step:u64::from(epoch)},native::FEATURE_TAG).unwrap();
            actual_float_policy();
        }
        let reference=context.into_completed_reference().unwrap();
        paired::validate_state(&trainer.weights,&reference,recipe).unwrap();
        let stage=native::SnapshotStage::AfterPosition{expected_step:3};
        let nearest=native::nearest_native03(&trainer.weights,recipe,stage,native::FEATURE_TAG).unwrap();
        let raw=native::encode_native03_layout(&nearest).unwrap();
        native::verify_native03_reexport(&trainer.weights,recipe,stage,native::FEATURE_TAG,&raw).unwrap();
        // The explicit memory diagnostic at step3 is not the production
        // checkpoint/native export, whose full-fit end remains step338043.
        assert!(checkpoint::completed_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
        assert!(checkpoint::completed_nearest_bytes(&trainer,recipe).is_err());
        assert!(checkpoint::initialized_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
        assert!(checkpoint::initialized_native_bytes(&trainer,recipe).is_err());
    }

    #[test]
    fn fullshape_synthetic_teacher_failure_poison_forbids_next_epoch_and_partial_export() {
        actual_float_policy();
        // Missing keyed value despite matching cardinality; both signed mate
        // thresholds and i32::MIN exercise the integer absolute-domain guard.
        for bad_teacher in [None,Some(30000_i32),Some(-30000_i32),Some(i32::MIN)] {
            let recipe=selected_recipe();
            let (mut trainer,reference)=checkpoint::fresh_trainer(recipe).unwrap();
            let mut context=Context::from_verified_declaration(reference,recipe).unwrap();
            let samples=[public_sample()];
            let good=ordinary_cache(&samples[0]);
            let token=trainer.train_paired_nonlinear_epoch(&samples,&good,1,&mut context).unwrap();
            assert_eq!(token.positions(),1); assert_eq!(token.end_step(),1);
            let before_failure=trainer.weights.clone();
            let bad=match bad_teacher {
                None=>HashMap::from([("public-synthetic-unmatched-key".to_string(),500)]),
                Some(cp)=>HashMap::from([(board_to_sfen(&samples[0].board),cp)]),
            };
            assert_eq!(bad.len(),samples.len());
            assert!(trainer.train_paired_nonlinear_epoch(&samples,&bad,2,&mut context).is_err());
            assert!(context.ensure_live().is_err());
            assert!(native::checkpoint_state_equal(&before_failure,&trainer.weights));
            assert_eq!(trainer.paired_optimizer_step(),1);
            // Repairing the cache cannot turn the terminal partial attempt
            // into a resumable one. No second update is allowed.
            assert!(trainer.train_paired_nonlinear_epoch(&samples,&good,2,&mut context).is_err());
            assert!(native::checkpoint_state_equal(&before_failure,&trainer.weights));
            assert!(checkpoint::completed_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
            assert!(checkpoint::completed_nearest_bytes(&trainer,recipe).is_err());
            assert!(checkpoint::initialized_checkpoint_bytes(&trainer,recipe,&memory_checkpoint_context()).is_err());
            assert!(checkpoint::initialized_native_bytes(&trainer,recipe).is_err());
            assert!(context.into_completed_reference().is_err());
            actual_float_policy();
        }
    }
}
