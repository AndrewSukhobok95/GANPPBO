import sys
import os

PROJECTDIR = os.getcwd()
GAN_DIR = os.path.join(PROJECTDIR, "base_modules/ganspace")
sys.path.insert(0, GAN_DIR)

import numpy as np
import pandas as pd
import ipywidgets as widgets

from jupGUI_utils.utils import verbose_info, widget_image_to_bytes

VERBOSE = False
VERBOSE_ENDL = "\t   "


class GUIses(object):
    def __init__(self,
                 ganpreffinder,ganpreffinder2):
        """
        :param ganpreffinder: GANPrefFinder class instance
        """
        self.optimize_Theta = False

        self.GANpf = ganpreffinder #for user model
        self.GANpf2 = ganpreffinder2 #for no user model

        WIDGET_CONTINIOUS_UPDATE = self.GANpf.USING_CUDA
        WIDGET_CONTINIOUS_UPDATE2 = self.GANpf2.USING_CUDA
        WIDGET_WIDTH = 300

        self.strength = 0
        self.strength2 = 0
        self.query_count = 1
        self.query_count2 = 1
        #self.w = self.GANpf.get_init_W()
        init_img = self.GANpf.get_init_img()

        self.X, self.Xi = self.GANpf.get_next_query()
        self.X2, self.Xi2 = self.GANpf2.get_next_query()  
        
        self.widget_image_init = widgets.Image(format='png', width=WIDGET_WIDTH,
                                               value=self.to_bytes(init_img))
        
        '''Panel with user model'''
        self.widget_image = widgets.Image(format='png', width=WIDGET_WIDTH,
                                          value=self.to_bytes(init_img))
        self.widget_pref_image = widgets.Image(format='png', width=WIDGET_WIDTH,
                                               value=self.to_bytes(init_img))
        self.widget_strength_slider = widgets.IntSlider(min=self.GANpf.left_bound,
                                                        max=self.GANpf.right_bound,
                                                        value=self.strength,
                                                        continuous_update=WIDGET_CONTINIOUS_UPDATE,
                                                        description='strength:')
        self.widget_strength_text = widgets.IntText(description="strength:",
                                                    continuous_update=WIDGET_CONTINIOUS_UPDATE)
        self.widget_button_incs = widgets.Button(description="slider +10 (both sides)",
                                                 buttin_style="info",
                                                 layout=widgets.Layout(width="150px"))
        self.widget_button_decs = widgets.Button(description="slider -10 (both sides)",
                                                 buttin_style="info",
                                                 layout=widgets.Layout(width="150px"))
        self.widget_button_nq = widgets.Button(description="next query",
                                               buttin_style="info",
                                               layout=widgets.Layout(width="300px"))
        self.queries_count_text = widgets.Label(value=str(self.query_count))
        self.widget_chb_adaptive_init = widgets.Checkbox(
            value=self.GANpf.ADAPTIVE_INITIALIZATION,
            description='DONT USE ACQUISITION FUNCTION', disabled=False, indent=False)
        
        
        '''Panel without user model'''
        self.widget_image2 = widgets.Image(format='png', width=WIDGET_WIDTH,
                                          value=self.to_bytes(init_img))
        self.widget_pref_image2 = widgets.Image(format='png', width=WIDGET_WIDTH,
                                               value=self.to_bytes(init_img))
        self.widget_strength_slider2 = widgets.IntSlider(min=self.GANpf2.left_bound,
                                                        max=self.GANpf2.right_bound,
                                                        value=self.strength2,
                                                        continuous_update=WIDGET_CONTINIOUS_UPDATE2,
                                                        description='strength:')
        self.widget_strength_text2 = widgets.IntText(description="strength:",
                                                    continuous_update=WIDGET_CONTINIOUS_UPDATE2)
        self.widget_button_incs2 = widgets.Button(description="slider +10 (both sides)",
                                                 buttin_style="info",
                                                 layout=widgets.Layout(width="150px"))
        self.widget_button_decs2 = widgets.Button(description="slider -10 (both sides)",
                                                 buttin_style="info",
                                                 layout=widgets.Layout(width="150px"))
        self.widget_button_nq2 = widgets.Button(description="next query",
                                               buttin_style="info",
                                               layout=widgets.Layout(width="300px"))
        self.queries_count_text2 = widgets.Label(value=str(self.query_count2))
        self.widget_chb_adaptive_init2 = widgets.Checkbox(
            value=self.GANpf2.ADAPTIVE_INITIALIZATION,
            description='DONT USE ACQUISITION FUNCTION', disabled=True, indent=False)
                                                                    
        
        ''' Sample preferred image '''
        self.widget_button_sample = widgets.Button(description="sample preferred image",
                                               buttin_style="info",
                                               layout=widgets.Layout(width="300px"))
        self.widget_sample_image = widgets.Image(format='png', width=WIDGET_WIDTH,
                                               value=self.to_bytes(init_img))


    def run(self):
        self.widget_strength_text.observe(self.change_strength, 'value')
        self.widget_button_nq.on_click(self.next_query)
        #self.widget_button_incs.on_click(self.increase_slider_range)
        #self.widget_button_decs.on_click(self.decrease_slider_range)
        self.widget_chb_adaptive_init.observe(self.switch_adaptive_init)
        # self.widget_chb_choose_comp.observe(self.switch_Xi_strategy)
        # self.widget_comp_slider.observe(self.change_component)
        widgets.link((self.widget_strength_slider, 'value'), (self.widget_strength_text, 'value'))
        manage_panel = widgets.VBox([
            widgets.Label(value='Control Panel:'),
            #widgets.HBox([self.widget_button_decs, self.widget_button_incs]),
            self.widget_strength_slider,
            self.widget_strength_text,
            self.widget_chb_adaptive_init,
            widgets.HBox([self.widget_button_nq,widgets.HBox([widgets.Label(value='Query no.'),self.queries_count_text])])
        ])
        image_panel = widgets.VBox([widgets.HTML(value="<font size=4><b>Image to Edit</b></font>"), self.widget_image])
        pref_image_panel = widgets.VBox([widgets.HTML(value="<font size=4><b>A Preferred Image (user model)</b></font>"), self.widget_pref_image])
        init_image_panel = widgets.VBox([widgets.HTML(value="<font size=4><b>Initial Image</b></font>"), self.widget_image_init])
        usermodel = widgets.VBox([
            widgets.HBox([image_panel, pref_image_panel, init_image_panel]),
            manage_panel
        ])
                                                                    
        self.widget_strength_text2.observe(self.change_strength2, 'value')
        self.widget_button_nq2.on_click(self.next_query2)
        self.widget_chb_adaptive_init2.observe(self.switch_adaptive_init2)
        widgets.link((self.widget_strength_slider2, 'value'), (self.widget_strength_text2, 'value'))
        manage_panel2 = widgets.VBox([
            widgets.Label(value='Control Panel:'),
            self.widget_strength_slider2,
            self.widget_strength_text2,
            self.widget_chb_adaptive_init2,
            widgets.HBox([self.widget_button_nq2,widgets.HBox([widgets.Label(value='Query no.'),self.queries_count_text2])])
        ])
        image_panel2 = widgets.VBox([widgets.HTML(value="<font size=4><b>Image to Edit</b></font>"), self.widget_image2])
        pref_image_panel2 = widgets.VBox([widgets.HTML(value="<font size=4><b>Preferred Image (no user model)</b></font>"), self.widget_pref_image2])
        nousermodel = widgets.VBox([
            widgets.HBox([image_panel2, pref_image_panel2, init_image_panel]),
            manage_panel2
        ])
                                                                                                                                 
        run_wid = widgets.VBox([usermodel,nousermodel])                                                      

        return run_wid
    
    
    def run_preferred_image_sampler(self):
        
        self.GANpf.prepare_random_Fourier_sampler()
        
        self.widget_button_sample.on_click(self.sample_pref_image)
        manage_panel = widgets.VBox([
            self.widget_button_sample
        ])
        sample_image_panel = widgets.VBox([widgets.HTML(value="<font size=4><b>Sampled image</b></font>"), self.widget_sample_image])
        run_wid = widgets.VBox([
            sample_image_panel,
            manage_panel
        ])
        
        return run_wid
    

    # === easy switching components ===

    def switch_adaptive_init(self, _):
        self.GANpf.switch_adaptive_initialization()
        if not self.GANpf.ADAPTIVE_INITIALIZATION:
            self.optimize_Theta = True
            
    def switch_adaptive_init2(self, _):
        pass

    def increase_slider_range(self, _):
        self.widget_strength_slider.min -= 10
        self.widget_strength_slider.max += 10

    def decrease_slider_range(self, _):
        self.widget_strength_slider.min += 10
        self.widget_strength_slider.max -= 10

    # === main functions ===

    def change_strength(self, strength):
        self._update_strength_value(strength.new)
        self._update_image()
    
    def change_strength2(self, strength):
        self._update_strength_value2(strength.new)
        self._update_image2()

    def next_query(self, _):
        self._updateGP()
        self._update_pref_image()
        # if self.optimize_Theta:
        #     print("+++ OPTIMIZING Theta for GP")
        #     self.ganpfinder.optimizeGP()
        #     self.optimize_Theta = False
        #     print("+++ FINISHED Theta OPTIMIZATION")
        self._update_X()
        self._next_query()
        self._update_strength_value(0)
        self._update_image()
        self._update_strength_slider_value()
        self.query_count += 1
        self.queries_count_text.value = str(self.query_count)
        
    def next_query2(self, _):
        self._update_pref_image2()
        self._update_X2()
        self._next_query2()
        self._update_strength_value2(0)
        self._update_image2()
        self._update_strength_slider_value2()
        self.query_count2 += 1
        self.queries_count_text2.value = str(self.query_count2)
        
    def sample_pref_image(self, _):
        self._update_sample_image()

    # === help functions ===

    @verbose_info(verbose=VERBOSE, msg="+ Updating GP", verbose_endl=VERBOSE_ENDL)
    def _updateGP(self):
        self.GANpf.updateGP(self.X, self.Xi, self.strength)

    @verbose_info(verbose=VERBOSE, msg="+ Getting next query (user model)", verbose_endl=VERBOSE_ENDL)
    def _next_query(self):
        self.X, self.Xi = self.GANpf.get_next_query()
    
    @verbose_info(verbose=VERBOSE, msg="+ Getting next query (no user model)", verbose_endl=VERBOSE_ENDL)
    def _next_query2(self):
        self.GANpf2.ADAPTIVE_INITIALIZATION=True
        self.X2, self.Xi2 = self.GANpf2.get_next_query()
        
    def _update_X(self):
        self.GANpf.update_adaptive_query(self.X, self.Xi, self.strength)
        
    def _update_X2(self):
        self.GANpf2.update_adaptive_query(self.X2, self.Xi2, self.strength2)

    @verbose_info(verbose=VERBOSE, msg="+ Updating image", verbose_endl=VERBOSE_ENDL)
    def _update_image(self):
        prefVec = self.GANpf.calculate_pref_vector(self.X, self.Xi, self.strength)
        layers_range = self.GANpf.get_comp_layers_range(self.Xi)
        img = self.GANpf.update_image(prefVec=prefVec, layers_range=layers_range)
        self.widget_image.value = self.to_bytes(img)
    
    @verbose_info(verbose=VERBOSE, msg="+ Updating image", verbose_endl=VERBOSE_ENDL)
    def _update_image2(self):
        prefVec = self.GANpf2.calculate_pref_vector(self.X2, self.Xi2, self.strength2)
        layers_range = self.GANpf2.get_comp_layers_range(self.Xi2)
        img = self.GANpf2.update_image(prefVec=prefVec, layers_range=layers_range)
        self.widget_image2.value = self.to_bytes(img)

    @verbose_info(verbose=VERBOSE, msg="+ Updating preferred image", verbose_endl=VERBOSE_ENDL)
    def _update_pref_image(self):
        X_star = self.GANpf.get_last_X_star_scaled()
        img = self.GANpf.update_image(prefVec=X_star)
        self.widget_pref_image.value = self.to_bytes(img)
        
    @verbose_info(verbose=VERBOSE, msg="+ Updating preferred image", verbose_endl=VERBOSE_ENDL)
    def _update_pref_image2(self):
        self.widget_pref_image2.value = self.widget_image2.value

    def _update_strength_slider_value(self):
        self.widget_strength_slider.value = self.strength
        
    def _update_strength_slider_value2(self):
        self.widget_strength_slider2.value = self.strength2

    @verbose_info(verbose=VERBOSE, msg="+ Updating strength parameter", verbose_endl=VERBOSE_ENDL)
    def _update_strength_value(self, new_value):
        self.strength = new_value
        
    @verbose_info(verbose=VERBOSE, msg="+ Updating strength parameter", verbose_endl=VERBOSE_ENDL)
    def _update_strength_value2(self, new_value):
        self.strength2 = new_value
        
    @verbose_info(verbose=VERBOSE, msg="+ Updating sample image", verbose_endl=VERBOSE_ENDL)
    def _update_sample_image(self):
        xstar = self.GANpf.sample_preferred_image()
        #print(xstar)
        #print(self.GANpf.GP_model.FP.unscale(self.GANpf.h_sampler.GP_xstar))
        img = self.GANpf.update_image(prefVec=xstar)
        self.widget_sample_image.value = self.to_bytes(img)



    # ===============================

    @staticmethod
    def to_bytes(img):
        return widget_image_to_bytes(img)

    # === next query ===

    # def _move_user_Xi(self):
    #     if self.ganpfinder.AQ.custom_Xi_strategy:
    #         if self.widget_comp_slider.value < self.widget_comp_slider.max:
    #             self.widget_comp_slider.value += 1
    #         else:
    #             self.widget_comp_slider.value += 1
    #
    # @verbose_info(verbose=VERBOSE, msg="+ Updating GUI parameters", verbose_endl=VERBOSE_ENDL)
    # def _updateGUI(self):
    #     self.strength = 0
    #     self._move_user_Xi()
    #     if self.start_with_init_img:
    #         img = self.ganpfinder.get_init_img()
    #     else:
    #         prefVec = self.ganpfinder.calculate_pref_vector(self.X, self.Xi, self.strength)
    #         img = self.ganpfinder.update_image(prefVec=prefVec)
    #     self.widget_image.value = self.to_bytes(img)
    #
    # # ==================
    #
    # # === custom change component ===
    #
    # def change_component(self, comp_index):
    #     comp_index_value = comp_index.new
    #     if type(comp_index_value) is int:
    #         self._Xi_from_user(comp_index_value)
    #         if self.ganpfinder.AQ.custom_Xi_strategy:
    #             self._update_image()
    #
    # @verbose_info(verbose=VERBOSE, msg="+ Updating component switch parameter", verbose_endl=VERBOSE_ENDL)
    # def _Xi_from_user(self, comp_index_value):
    #     self.ganpfinder.Xi_from_user(comp_index_value)
    #     self.Xi = self.ganpfinder.AQ.get_Xi()
    #
    # @verbose_info(verbose=VERBOSE, msg="+ Updating GUI parameters", verbose_endl=VERBOSE_ENDL)
    # def _update_image(self):
    #     img = self._modify_by_strength(self.strength)
    #     self.widget_image.value = self.to_bytes(img)

