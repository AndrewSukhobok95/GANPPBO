# import sys
# import os
# from datetime import datetime
# import numpy as np
# import pandas as pd
# from PIL import Image
#
# import ipywidgets as widgets
# from io import BytesIO
#
# from jupGUI_utils.utils import verbose_info
#
# # s = widgets.IntSlider()
# # b = widgets.Button(description="Click Me!")
# # def on_button_clicked(_):
# #     s.max += 10
# # b.on_click(on_button_clicked)
# # display(s, b)
#
# VERBOSE = True
# VERBOSE_ENDL = "   "
#
#
# class GUIses(object):
#     def __init__(self, ganpfinder, start_with_init_img=False):
#         self.start_with_init_img = start_with_init_img
#
#         self.ganpfinder = ganpfinder
#
#         self.strength = 0
#         self.w = self.ganpfinder.get_init_W()
#         init_img = self.ganpfinder.get_init_img()
#         self.img = init_img
#
#         self.X, self.Xi = self.ganpfinder.get_next_query()
#
#         self.widget_strength_slider = widgets.IntSlider(min=-20, max=20,
#                                                         value=self.strength,
#                                                         continuous_update=False,
#                                                         description='strength:')
#
#         self.widget_strength_text = widgets.IntText(description="strength:",
#                                                     continuous_update=False)
#
#         self.widget_button_incs = widgets.Button(description="slider +10 (both sides)",
#                                                  buttin_style="info",
#                                                  layout=widgets.Layout(width="150px"))
#
#         self.widget_button_decs = widgets.Button(description="slider -10 (both sides)",
#                                                  buttin_style="info",
#                                                  layout=widgets.Layout(width="150px"))
#
#         self.widget_button_nq = widgets.Button(description="next query",
#                                                buttin_style="info",
#                                                layout=widgets.Layout(width="300px"))
#
#         self.widget_chb_adaptive_init = widgets.Checkbox(
#             value=self.ganpfinder.ADAPTIVE_INITIALIZATION,
#             description='adaptive initialization', disabled=False, indent=False)
#
#         self.widget_chb_choose_comp = widgets.Checkbox(
#             value=self.ganpfinder.AQ.custom_Xi_strategy,
#             description='choose component', disabled=False, indent=False)
#
#         self.current_comp_index = 0
#         self.widget_comp_slider = widgets.IntSlider(min=0,
#                                                     max=self.ganpfinder.N_comp_in_use-1,
#                                                     value=self.current_comp_index,
#                                                     continuous_update=False,
#                                                     description='component:')
#
#         self.widget_image = widgets.Image(format='png', width=300,
#                                           value=self.to_bytes(init_img))
#         self.widget_image_init = widgets.Image(format='png', width=300,
#                                                value=self.to_bytes(init_img))
#
#     def run(self):
#         self.widget_strength_text.observe(self.change_strength, 'value')
#         # self.widget_strength_slider.observe(self.change_strength, 'value')
#         self.widget_button_nq.on_click(self.next_query)
#         self.widget_button_incs.on_click(self.increase_slider_range)
#         self.widget_button_decs.on_click(self.decrease_slider_range)
#         self.widget_chb_adaptive_init.observe(self.switch_adaptive_init)
#         self.widget_chb_choose_comp.observe(self.switch_Xi_strategy)
#         self.widget_comp_slider.observe(self.change_component)
#
#         widgets.link((self.widget_strength_slider, 'value'), (self.widget_strength_text, 'value'))
#
#         manage_panel = widgets.VBox([
#             widgets.HBox([self.widget_button_decs, self.widget_button_incs]),
#             self.widget_strength_slider,
#             self.widget_strength_text,
#             self.widget_chb_adaptive_init,
#             self.widget_chb_choose_comp,
#             self.widget_comp_slider,
#             self.widget_button_nq
#         ])
#
#         run_wid = widgets.HBox([
#             manage_panel,
#             self.widget_image,
#             self.widget_image_init
#         ])
#
#         return run_wid
#
#     # === easy switching components ===
#
#     def switch_adaptive_init(self, _):
#         self.ganpfinder.switch_adaptive_initialization()
#
#     def switch_Xi_strategy(self, _):
#         self.ganpfinder.switch_Xi_strategy()
#
#     def increase_slider_range(self, _):
#         self.widget_strength_slider.min -= 10
#         self.widget_strength_slider.max += 10
#
#     def decrease_slider_range(self, _):
#         self.widget_strength_slider.min += 10
#         self.widget_strength_slider.max -= 10
#
#     # =================================
#
#     # === change strength ===
#
#     @verbose_info(verbose=VERBOSE, msg="+ Updating strength parameter", verbose_endl=VERBOSE_ENDL)
#     def change_strength(self, strength):
#         img = self._modify_by_strength(strength.new)
#         self.widget_image.value = self.to_bytes(img)
#
#     def _modify_by_strength(self, strength):
#         self.strength = strength
#         prefVec = self.ganpfinder.calculate_pref_vector(self.X, self.Xi, self.strength)
#         img = self.ganpfinder._update_image(prefVec=prefVec)
#         return img
#
#     # ======================
#
#     # === next query ===
#
#     def next_query(self, _):
#         self._updateGP()
#         self._next_query()
#         self._updateGUI()
#         self.widget_strength_slider.value = self.strength
#
#     @verbose_info(verbose=VERBOSE, msg="+ Updating GP", verbose_endl=VERBOSE_ENDL)
#     def _updateGP(self):
#         self.ganpfinder.updateGP(self.X, self.Xi, self.strength)
#
#     @verbose_info(verbose=VERBOSE, msg="+ Getting next query", verbose_endl=VERBOSE_ENDL)
#     def _next_query(self):
#         self.X, self.Xi = self.ganpfinder.get_next_query()
#
#     def _move_user_Xi(self):
#         if self.ganpfinder.AQ.custom_Xi_strategy:
#             if self.widget_comp_slider.value < self.widget_comp_slider.max:
#                 self.widget_comp_slider.value += 1
#             else:
#                 self.widget_comp_slider.value += 1
#
#     @verbose_info(verbose=VERBOSE, msg="+ Updating GUI parameters", verbose_endl=VERBOSE_ENDL)
#     def _updateGUI(self):
#         self.strength = 0
#         self._move_user_Xi()
#         if self.start_with_init_img:
#             img = self.ganpfinder.get_init_img()
#         else:
#             prefVec = self.ganpfinder.calculate_pref_vector(self.X, self.Xi, self.strength)
#             img = self.ganpfinder._update_image(prefVec=prefVec)
#         self.widget_image.value = self.to_bytes(img)
#
#     # ==================
#
#     # === custom change component ===
#
#     def change_component(self, comp_index):
#         comp_index_value = comp_index.new
#         if type(comp_index_value) is int:
#             self._Xi_from_user(comp_index_value)
#             if self.ganpfinder.AQ.custom_Xi_strategy:
#                 self._update_image()
#
#     @verbose_info(verbose=VERBOSE, msg="+ Updating component switch parameter", verbose_endl=VERBOSE_ENDL)
#     def _Xi_from_user(self, comp_index_value):
#         self.ganpfinder.Xi_from_user(comp_index_value)
#         self.Xi = self.ganpfinder.AQ.get_Xi()
#
#     @verbose_info(verbose=VERBOSE, msg="+ Updating GUI parameters", verbose_endl=VERBOSE_ENDL)
#     def _update_image(self):
#         img = self._modify_by_strength(self.strength)
#         self.widget_image.value = self.to_bytes(img)
#
#     # ===============================
#
#     @staticmethod
#     def to_bytes(img):
#         image = Image.fromarray((img * 255).astype(np.uint8), mode='RGB')
#         f = BytesIO()
#         image.save(f, 'png')
#         return f.getvalue()
