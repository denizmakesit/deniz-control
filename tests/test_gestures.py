"""Offline unit tests for gesture logic and cursor mapping. No camera, no real mouse.

    python -m unittest discover -s tests -p "test_gestures.py" -v
"""

import unittest

import numpy as np

from deniz_control.config import GestureConfig, MouseConfig
from deniz_control.gestures import (GestureState, GestureStateMachine, HandFeatures, Pose,
                                    extract_features)
from deniz_control.mouse_controller import ExponentialSmoother, MouseController, map_to_screen


def synthetic_hand(index_up=True, others_up=False, pinch=False) -> np.ndarray:
    """Build a simple upright hand in normalized coords (aspect 1.0).

    Wrist at the bottom, fingers pointing up (smaller y). Folded fingers place
    the tip back near the knuckle.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.50, 0.80, 0)                      # wrist
    lm[1:5] = [(0.42, 0.75, 0), (0.37, 0.70, 0), (0.34, 0.65, 0), (0.32, 0.60, 0)]  # thumb
    bases = {5: 0.45, 9: 0.50, 13: 0.55, 17: 0.60}   # MCP x for index/middle/ring/pinky
    for mcp, x in bases.items():
        up = index_up if mcp == 5 else others_up
        lm[mcp] = (x, 0.60, 0)
        lm[mcp + 1] = (x, 0.52, 0)               # PIP
        if up:
            lm[mcp + 2] = (x, 0.46, 0)
            lm[mcp + 3] = (x, 0.40, 0)           # tip
        else:
            lm[mcp + 2] = (x, 0.56, 0)
            lm[mcp + 3] = (x, 0.62, 0)           # tip curled back below PIP
    if pinch:
        lm[4] = lm[8] + np.array([0.01, 0.01, 0], dtype=np.float32)
    return lm


def feats(pointing=True, pinch_ratio=1.0, v=False, y=0.4) -> HandFeatures:
    """Synthetic features. v=True gives the V-sign; y moves the fingertips."""
    return HandFeatures(pointing or v, v, False, False, pinch_ratio, (0.5, y), (0.55, y))


class FeatureTests(unittest.TestCase):
    cfg = GestureConfig()

    def test_pointing_pose(self):
        f = extract_features(synthetic_hand(index_up=True), 1.0, self.cfg)
        self.assertTrue(f.is_pointing)
        self.assertGreater(f.pinch_ratio, self.cfg.pinch_off_ratio)

    def test_open_hand_is_not_pointing(self):
        f = extract_features(synthetic_hand(index_up=True, others_up=True), 1.0, self.cfg)
        self.assertFalse(f.is_pointing)

    def test_v_sign_pose(self):
        lm = synthetic_hand(index_up=True)
        lm[11], lm[12] = (0.50, 0.46, 0), (0.50, 0.40, 0)    # extend middle finger
        self.assertEqual(extract_features(lm, 1.0, self.cfg).pose, Pose.V)

    def test_fist_is_not_pointing(self):
        f = extract_features(synthetic_hand(index_up=False), 1.0, self.cfg)
        self.assertFalse(f.is_pointing)

    def test_pinch_ratio_small_when_pinching(self):
        f = extract_features(synthetic_hand(pinch=True), 1.0, self.cfg)
        self.assertLess(f.pinch_ratio, self.cfg.pinch_on_ratio)

    def test_pinch_ratio_is_scale_invariant(self):
        a = synthetic_hand()
        b = a.copy()
        b[:, :2] = 0.5 + (b[:, :2] - 0.5) * 0.5     # same hand, half the size
        ra = extract_features(a, 1.0, self.cfg).pinch_ratio
        rb = extract_features(b, 1.0, self.cfg).pinch_ratio
        self.assertAlmostEqual(ra, rb, places=4)


PRESSED = (GestureState.CLICKING, GestureState.DRAGGING)


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = GestureConfig()
        self.sm = GestureStateMachine(self.cfg)
        self.t = 0.0
        self.events = []

    def step(self, f, dt=1 / 30):
        self.t += dt
        return self.sm.update(f, self.t)

    def run_for(self, f, seconds):
        upd, clicks = None, 0
        for _ in range(int(seconds * 30)):
            upd = self.step(f)
            clicks += upd.click
            self.events.append(upd)
        return upd, clicks

    events: list

    def count(self, attr):
        return sum(bool(getattr(u, attr)) for u in self.events)

    def test_idle_without_hand(self):
        self.assertEqual(self.step(None).state, GestureState.IDLE)

    def test_pointing_needs_confirmation(self):
        self.assertEqual(self.step(feats()).state, GestureState.TRACKING)
        upd, _ = self.run_for(feats(), 0.3)
        self.assertEqual(upd.state, GestureState.MOVING)

    def test_one_click_per_pinch(self):
        self.run_for(feats(), 0.3)
        upd, clicks = self.run_for(feats(pinch_ratio=0.1), 1.0)   # hold pinch 1 s
        self.assertIn(upd.state, PRESSED)      # long hold turns into a drag
        self.assertEqual(clicks, 1)

    def test_hysteresis_band_does_not_release(self):
        self.run_for(feats(), 0.3)
        self.run_for(feats(pinch_ratio=0.1), 0.2)
        # 0.38 is between on (0.30) and off (0.45): must stay pinched, no new click
        upd, clicks = self.run_for(feats(pinch_ratio=0.38), 0.5)
        self.assertIn(upd.state, PRESSED)
        self.assertEqual(clicks, 0)

    def test_release_then_second_click_after_cooldown(self):
        self.run_for(feats(), 0.3)
        _, c1 = self.run_for(feats(pinch_ratio=0.1), 0.2)
        upd, _ = self.run_for(feats(pinch_ratio=0.8), 0.5)
        self.assertEqual(upd.state, GestureState.MOVING)
        _, c2 = self.run_for(feats(pinch_ratio=0.1), 0.2)
        self.assertEqual((c1, c2), (1, 1))

    def test_cooldown_blocks_fast_repeat(self):
        self.sm = GestureStateMachine(GestureConfig(click_cooldown_s=0.5))
        self.run_for(feats(), 0.3)
        total = 0
        for _ in range(3):   # pinch/release cycles faster than the 0.4 s cooldown
            _, c = self.run_for(feats(pinch_ratio=0.1), 0.1)
            total += c
            self.run_for(feats(pinch_ratio=0.8), 0.1)
        self.assertLess(total, 3)

    def test_single_frame_pinch_glitch_does_not_click(self):
        self.run_for(feats(), 0.3)
        upd = self.step(feats(pinch_ratio=0.1))
        self.assertFalse(upd.click)
        upd = self.step(feats(pinch_ratio=0.9))
        self.assertEqual(upd.state, GestureState.MOVING)

    def test_no_click_without_pointing_first(self):
        _, clicks = self.run_for(feats(pointing=False, pinch_ratio=0.1), 1.0)
        self.assertEqual(clicks, 0)

    def test_every_press_is_released(self):
        self.run_for(feats(), 0.3)
        self.run_for(feats(pinch_ratio=0.1), 0.2)
        self.run_for(feats(pinch_ratio=0.8), 0.3)
        self.assertEqual(self.count("left_down"), 1)
        self.assertEqual(self.count("left_up"), 1)

    def test_long_pinch_becomes_drag(self):
        self.run_for(feats(), 0.3)
        upd, _ = self.run_for(feats(pinch_ratio=0.1), 0.1)
        self.assertEqual(upd.state, GestureState.CLICKING)
        upd, _ = self.run_for(feats(pinch_ratio=0.1), 0.5)
        self.assertEqual(upd.state, GestureState.DRAGGING)
        upd, _ = self.run_for(feats(pinch_ratio=0.8), 0.3)
        self.assertEqual(upd.state, GestureState.MOVING)
        self.assertEqual((self.count("left_down"), self.count("left_up")), (1, 1))

    def test_hand_lost_while_dragging_releases_button(self):
        self.run_for(feats(), 0.3)
        self.run_for(feats(pinch_ratio=0.1), 0.6)
        self.assertTrue(self.sm.button_down)
        upd, _ = self.run_for(None, 0.5)
        self.assertEqual(upd.state, GestureState.IDLE)
        self.assertFalse(self.sm.button_down)
        self.assertEqual(self.count("left_up"), 1)

    def test_v_sign_scrolls_with_hand_movement(self):
        upd, _ = self.run_for(feats(v=True, y=0.4), 0.3)
        self.assertEqual(upd.state, GestureState.SCROLLING)
        total = 0.0
        for i in range(30):                     # move hand down over 1 s
            total += self.step(feats(v=True, y=0.4 + 0.2 * (i + 1) / 30)).scroll_dy
        for _ in range(30):                     # hold still, let smoothing settle
            total += self.step(feats(v=True, y=0.6)).scroll_dy
        self.assertAlmostEqual(total, 0.2, delta=0.01)
        self.assertEqual(self.count("left_down"), 0)

    def test_v_sign_pinch_is_one_right_click(self):
        self.run_for(feats(v=True), 0.3)
        upd, _ = self.run_for(feats(v=True, pinch_ratio=0.1), 0.8)
        self.assertEqual(upd.state, GestureState.RIGHT_CLICK)
        self.assertEqual(self.count("right_click"), 1)
        self.assertEqual(self.count("left_down"), 0)
        upd, _ = self.run_for(feats(v=True), 0.3)
        self.assertEqual(upd.state, GestureState.SCROLLING)

    def test_pinch_held_before_pointing_does_not_click(self):
        # pinch is already closed when the pose is confirmed -> must release first
        self.run_for(feats(pointing=False, pinch_ratio=0.1), 0.3)
        _, clicks = self.run_for(feats(pinch_ratio=0.1), 0.5)
        self.assertEqual(clicks, 0)

    def test_open_palm_starts_and_stops_listening(self):
        palm = HandFeatures(True, True, True, True, 1.0, (0.5, 0.4), (0.55, 0.4))
        upd, _ = self.run_for(palm, 0.2)
        self.assertNotEqual(upd.state, GestureState.LISTENING)   # not held long enough
        upd, _ = self.run_for(palm, 0.3)
        self.assertEqual(upd.state, GestureState.LISTENING)
        upd, _ = self.run_for(feats(), 0.3)                      # back to pointing
        self.assertEqual(upd.state, GestureState.MOVING)
        self.assertEqual(self.count("left_down"), 0)

    def test_brief_dropout_keeps_state(self):
        self.run_for(feats(), 0.3)
        self.assertEqual(self.step(None).state, GestureState.MOVING)
        upd, _ = self.run_for(None, 0.5)
        self.assertEqual(upd.state, GestureState.IDLE)


class FakeBackend:
    def __init__(self):
        self.moves, self.buttons, self.right, self.scrolled = [], [], 0, 0
        self.pos = (500, 500)

    def size(self): return (1710, 1107)
    def position(self): return self.pos
    def fail_safe_check(self): pass
    def double_click_interval(self): return 0.5

    def move(self, x, y, dragging):
        self.moves.append((x, y, dragging))
        self.pos = (x, y)

    def left(self, down, x, y, click_count):
        self.buttons.append((down, click_count))

    def right_click(self, x, y): self.right += 1
    def scroll(self, pixels): self.scrolled += pixels


class MouseTests(unittest.TestCase):
    def test_mapping_corners_and_margin(self):
        cfg = MouseConfig()
        l, t, r, b = cfg.active_region
        m = cfg.screen_edge_margin_px
        self.assertEqual(map_to_screen(0.0, 0.0, 1710, 1107, cfg), (m, m))
        self.assertEqual(map_to_screen(1.0, 1.0, 1710, 1107, cfg), (1709 - m, 1106 - m))
        cx, cy = map_to_screen((l + r) / 2, (t + b) / 2, 1710, 1107, cfg)
        self.assertAlmostEqual(cx, 1709 / 2)
        self.assertAlmostEqual(cy, 1106 / 2)

    def test_smoother_converges_and_ignores_jitter(self):
        s = ExponentialSmoother(alpha=0.35, deadzone=1.5)
        s.update(0, 0)
        for _ in range(40):
            x, _ = s.update(100, 0)
        self.assertAlmostEqual(x, 100, delta=2)
        before = s.value
        s.update(before[0] + 1, before[1])   # sub-deadzone wiggle
        self.assertEqual(s.value, before)

    def test_no_duplicate_moves(self):
        os_ = FakeBackend()
        mc = MouseController(MouseConfig(), backend=os_)
        mc.move_towards(0.5, 0.4)
        mc.move_towards(0.5, 0.4)
        self.assertEqual(len(os_.moves), 1)

    def test_drag_sends_dragged_events_and_anchor_prevents_jump(self):
        os_ = FakeBackend()
        mc = MouseController(MouseConfig(smoothing_alpha=1.0), backend=os_)
        start = mc.move_towards(0.5, 0.4)
        mc.left_down()
        mc.anchor(0.52, 0.42)                 # fingertip shifted while pinching
        self.assertEqual(mc.move_towards(0.52, 0.42), start)   # no jump
        mc.move_towards(0.60, 0.42)
        self.assertTrue(os_.moves[-1][2])     # dragged event while button held
        mc.left_up()
        self.assertEqual([b[0] for b in os_.buttons], [True, False])

    def test_double_click_count(self):
        os_ = FakeBackend()
        mc = MouseController(MouseConfig(), backend=os_)
        mc.click()
        mc.click()
        self.assertEqual([b[1] for b in os_.buttons], [1, 1, 2, 2])

    def test_release_all_lifts_held_button(self):
        os_ = FakeBackend()
        mc = MouseController(MouseConfig(), backend=os_)
        mc.left_down()
        mc.release_all()
        mc.release_all()                       # idempotent
        self.assertEqual([b[0] for b in os_.buttons], [True, False])
        self.assertFalse(mc.left_is_down)

    def test_slow_scroll_accumulates(self):
        os_ = FakeBackend()
        mc = MouseController(MouseConfig(scroll_speed=1.0), backend=os_)
        for _ in range(100):
            mc.scroll(0.0001)                  # 0.11 px each -> adds up
            mc.reset()                         # app resets cursor smoothing every frame
        self.assertGreaterEqual(os_.scrolled, 10)


if __name__ == "__main__":
    unittest.main()
