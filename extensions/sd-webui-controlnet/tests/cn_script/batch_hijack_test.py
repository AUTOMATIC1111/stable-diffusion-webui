import unittest.mock
import importlib
from typing import Any

utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')


from modules import processing, scripts, shared
from scripts import controlnet, external_code, batch_hijack


batch_hijack.instance.undo_hijack()
original_process_images_inner = processing.process_images_inner


class TestBatchHijack(unittest.TestCase):
    @unittest.mock.patch('modules.script_callbacks.on_script_unloaded')
    def setUp(self, on_script_unloaded_mock):
        self.on_script_unloaded_mock = on_script_unloaded_mock

        self.batch_hijack_object = batch_hijack.BatchHijack()
        self.batch_hijack_object.do_hijack()

    def tearDown(self):
        self.batch_hijack_object.undo_hijack()

    def test_do_hijack__registers_on_script_unloaded(self):
        self.on_script_unloaded_mock.assert_called_once_with(self.batch_hijack_object.undo_hijack)

    def test_do_hijack__call_once__hijacks_once(self):
        self.assertEqual(getattr(processing, '__controlnet_original_process_images_inner'), original_process_images_inner)
        self.assertEqual(processing.process_images_inner, self.batch_hijack_object.processing_process_images_hijack)

    @unittest.mock.patch('modules.processing.__controlnet_original_process_images_inner')
    def test_do_hijack__multiple_times__hijacks_once(self, process_images_inner_mock):
        self.batch_hijack_object.do_hijack()
        self.batch_hijack_object.do_hijack()
        self.batch_hijack_object.do_hijack()
        self.assertEqual(process_images_inner_mock, getattr(processing, '__controlnet_original_process_images_inner'))


class TestGetControlNetBatchesWorks(unittest.TestCase):
    def setUp(self):
        self.p = unittest.mock.MagicMock()
        assert scripts.scripts_txt2img is not None
        self.p.scripts = scripts.scripts_txt2img
        self.cn_script = controlnet.Script()
        self.p.scripts.alwayson_scripts = [self.cn_script]
        self.p.script_args = []

    def tearDown(self):
        batch_hijack.instance.dispatch_callbacks(batch_hijack.instance.postprocess_batch_callbacks, self.p)

    def assert_get_cn_batches_works(self, batch_images_list):
        self.cn_script.args_from = 0
        self.cn_script.args_to = self.cn_script.args_from + len(self.p.script_args)

        is_cn_batch, batches, output_dir, _ = batch_hijack.get_cn_batches(self.p)
        batch_hijack.instance.dispatch_callbacks(batch_hijack.instance.process_batch_callbacks, self.p, batches, output_dir)

        batch_units = [unit for unit in self.p.script_args if getattr(unit, 'input_mode', batch_hijack.InputMode.SIMPLE) == batch_hijack.InputMode.BATCH]
        if batch_units:
            self.assertEqual(min(len(list(unit.batch_images)) for unit in batch_units), len(batches))
        else:
            self.assertEqual(1, len(batches))

        for i, unit in enumerate(self.cn_script.enabled_units):
            self.assertListEqual(batch_images_list[i], list(unit.batch_images))

    def test_get_cn_batches__empty(self):
        is_batch, batches, _, _ = batch_hijack.get_cn_batches(self.p)
        self.assertEqual(1, len(batches))
        self.assertEqual(is_batch, False)

    def test_get_cn_batches__1_simple(self):
        self.p.script_args.append(external_code.ControlNetUnit(image=get_dummy_image()))
        self.assert_get_cn_batches_works([
            [self.p.script_args[0].image],
        ])

    def test_get_cn_batches__2_simples(self):
        self.p.script_args.extend([
            external_code.ControlNetUnit(image=get_dummy_image(0)),
            external_code.ControlNetUnit(image=get_dummy_image(1)),
        ])
        self.assert_get_cn_batches_works([
            [get_dummy_image(0)],
            [get_dummy_image(1)],
        ])

    def test_get_cn_batches__1_batch(self):
        self.p.script_args.extend([
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(0),
                    get_dummy_image(1),
                ],
            ),
        ])
        self.assert_get_cn_batches_works([
            [
                get_dummy_image(0),
                get_dummy_image(1),
            ],
        ])

    def test_get_cn_batches__2_batches(self):
        self.p.script_args.extend([
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(0),
                    get_dummy_image(1),
                ],
            ),
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(2),
                    get_dummy_image(3),
                ],
            ),
        ])
        self.assert_get_cn_batches_works([
            [
                get_dummy_image(0),
                get_dummy_image(1),
            ],
            [
                get_dummy_image(2),
                get_dummy_image(3),
            ],
        ])

    def test_get_cn_batches__2_mixed(self):
        self.p.script_args.extend([
            external_code.ControlNetUnit(image=get_dummy_image(0)),
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(1),
                    get_dummy_image(2),
                ],
            ),
        ])
        self.assert_get_cn_batches_works([
            [
                get_dummy_image(0),
                get_dummy_image(0),
            ],
            [
                get_dummy_image(1),
                get_dummy_image(2),
            ],
        ])

    def test_get_cn_batches__3_mixed(self):
        self.p.script_args.extend([
            external_code.ControlNetUnit(image=get_dummy_image(0)),
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(1),
                    get_dummy_image(2),
                    get_dummy_image(3),
                ],
            ),
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(4),
                    get_dummy_image(5),
                ],
            ),
        ])
        self.assert_get_cn_batches_works([
            [
                get_dummy_image(0),
                get_dummy_image(0),
            ],
            [
                get_dummy_image(1),
                get_dummy_image(2),
            ],
            [
                get_dummy_image(4),
                get_dummy_image(5),
            ],
        ])

class TestProcessImagesPatchWorks(unittest.TestCase):
    @unittest.mock.patch('modules.script_callbacks.on_script_unloaded')
    def setUp(self, on_script_unloaded_mock):
        self.on_script_unloaded_mock = on_script_unloaded_mock
        self.p = unittest.mock.MagicMock()
        assert scripts.scripts_txt2img is not None
        self.p.scripts = scripts.scripts_txt2img
        self.cn_script = controlnet.Script()
        self.p.scripts.alwayson_scripts = [self.cn_script]
        self.p.script_args = []
        self.p.all_seeds = [0]
        self.p.all_subseeds = [0]
        self.old_model, shared.sd_model = shared.sd_model, unittest.mock.MagicMock()

        self.batch_hijack_object = batch_hijack.BatchHijack()
        self.callbacks_mock = unittest.mock.MagicMock()
        self.batch_hijack_object.process_batch_callbacks.append(self.callbacks_mock.process)
        self.batch_hijack_object.process_batch_each_callbacks.append(self.callbacks_mock.process_each)
        self.batch_hijack_object.postprocess_batch_each_callbacks.insert(0, self.callbacks_mock.postprocess_each)
        self.batch_hijack_object.postprocess_batch_callbacks.insert(0, self.callbacks_mock.postprocess)
        self.batch_hijack_object.do_hijack()
        shared.state.begin()

    def tearDown(self):
        shared.state.end()
        self.batch_hijack_object.undo_hijack()
        shared.sd_model = self.old_model

    @unittest.mock.patch('modules.processing.__controlnet_original_process_images_inner')
    def assert_process_images_hijack_called(self, process_images_mock, batch_count):
        process_images_mock.return_value = processing.Processed(self.p, [get_dummy_image('output')])
        with unittest.mock.patch.dict(shared.opts.data, {
            'controlnet_show_batch_images_in_ui': True,
        }):
            res = processing.process_images_inner(self.p)

        self.assertEqual(res, process_images_mock.return_value)

        if batch_count > 0:
            self.callbacks_mock.process.assert_called()
            self.callbacks_mock.postprocess.assert_called()
        else:
            self.callbacks_mock.process.assert_not_called()
            self.callbacks_mock.postprocess.assert_not_called()

        self.assertEqual(self.callbacks_mock.process_each.call_count, batch_count)
        self.assertEqual(self.callbacks_mock.postprocess_each.call_count, batch_count)

    def test_process_images_no_units_forwards(self):
        self.assert_process_images_hijack_called(batch_count=0)

    def test_process_images__only_simple_units__forwards(self):
        self.p.script_args = [
            external_code.ControlNetUnit(image=get_dummy_image()),
            external_code.ControlNetUnit(image=get_dummy_image()),
        ]
        self.assert_process_images_hijack_called(batch_count=0)

    def test_process_images__1_batch_1_unit__runs_1_batch(self):
        self.p.script_args = [
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(),
                ],
            ),
        ]
        self.assert_process_images_hijack_called(batch_count=1)

    def test_process_images__2_batches_1_unit__runs_2_batches(self):
        self.p.script_args = [
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(0),
                    get_dummy_image(1),
                ],
            ),
        ]
        self.assert_process_images_hijack_called(batch_count=2)

    def test_process_images__8_batches_1_unit__runs_8_batches(self):
        batch_count = 8
        self.p.script_args = [
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[get_dummy_image(i) for i in range(batch_count)]
            ),
        ]
        self.assert_process_images_hijack_called(batch_count=batch_count)

    def test_process_images__1_batch_2_units__runs_1_batch(self):
        self.p.script_args = [
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[get_dummy_image(0)]
            ),
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[get_dummy_image(1)]
            ),
        ]
        self.assert_process_images_hijack_called(batch_count=1)

    def test_process_images__2_batches_2_units__runs_2_batches(self):
        self.p.script_args = [
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(0),
                    get_dummy_image(1),
                ],
            ),
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(2),
                    get_dummy_image(3),
                ],
            ),
        ]
        self.assert_process_images_hijack_called(batch_count=2)

    def test_process_images__3_batches_2_mixed_units__runs_3_batches(self):
        self.p.script_args = [
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.BATCH,
                batch_images=[
                    get_dummy_image(0),
                    get_dummy_image(1),
                    get_dummy_image(2),
                ],
            ),
            controlnet.UiControlNetUnit(
                input_mode=batch_hijack.InputMode.SIMPLE,
                image=get_dummy_image(3),
            ),
        ]
        self.assert_process_images_hijack_called(batch_count=3)


def get_dummy_image(name: Any = 0):
    return f'base64#{name}...'

if __name__ == '__main__':
    unittest.main()