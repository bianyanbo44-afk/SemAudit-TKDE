"""Author the scientific mechanism artwork as editable vector groups for Illustrator."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import xml.etree.ElementTree as ET
from PIL import ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1000, 790
WIDTH_MM = 183
SCALE_PT = WIDTH_MM / 25.4 * 72 / W
INK = '#203448'
MUTED = '#5F7182'
LINE = '#A8B8C7'
BLUE = '#3A6F9D'
TEAL = '#237D80'
ORANGE = '#C68149'
PALETTE = {
    'u': ('#DDEAF5', '#88A9C7', '#2F5F87'),
    'v': ('#F5E4D5', '#D0A27B', '#986033'),
    'w': ('#E0EEE6', '#91B09E', '#3D7057'),
    'r': ('#E9E2F2', '#AE9AC5', '#76598F'),
    's': ('#F2ECD7', '#C3B47F', '#82713A'),
    '0': ('#EEF2F5', '#CCD6DF', '#6B7B88'),
}
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ROOT = ET.Element('{' + NS + '}svg', {
    'width': f'{WIDTH_MM}mm', 'height': f'{WIDTH_MM * H / W}mm',
    'viewBox': f'0 0 {W} {H}', 'version': '1.1',
})
ET.SubElement(ROOT, '{' + NS + '}title').text = 'Representation-specific semantic audit'
ET.SubElement(ROOT, '{' + NS + '}desc').text = (
    'Schematic mechanism for the TKDE manuscript. Matched frozen text representations, '
    'event-preserving content interventions, independently selected condition models, '
    'fixed aligned checkpoints and known-signal calibration. All traces and vector '
    'symbols are conceptual illustrations.'
)
PARENT = [ROOT]
TEXT_RECORDS = []


def element(tag, **attributes):
    return ET.SubElement(PARENT[-1], '{' + NS + '}' + tag,
                         {key.replace('_', '-'): str(value) for key, value in attributes.items()})


@contextmanager
def group(name):
    node = element('g', id=name)
    PARENT.append(node)
    yield
    PARENT.pop()


def rect(x, y, width, height, fill='none', stroke='none', radius=0, weight=1):
    return element('rect', x=x, y=y, width=width, height=height, rx=radius,
                   fill=fill, stroke=stroke, stroke_width=weight)


def circle(x, y, radius, fill, stroke='none', weight=1):
    return element('circle', cx=x, cy=y, r=radius, fill=fill, stroke=stroke, stroke_width=weight)


def path(points, color=LINE, weight=1.5, dash=None, arrow=False):
    attributes = dict(d='M ' + ' L '.join(f'{x:g},{y:g}' for x, y in points),
                      fill='none', stroke=color, stroke_width=weight,
                      stroke_linecap='round', stroke_linejoin='round')
    if dash:
        attributes['stroke_dasharray'] = dash
    element('path', **attributes)
    if arrow:
        x, y = points[-1]
        start_x, start_y = points[-2]
        angle = math.atan2(y - start_y, x - start_x)
        size = 7.0
        back_x, back_y = x - size * math.cos(angle), y - size * math.sin(angle)
        normal_x, normal_y = -math.sin(angle) * 3.2, math.cos(angle) * 3.2
        vertices = [(x, y), (back_x + normal_x, back_y + normal_y),
                    (back_x - normal_x, back_y - normal_y)]
        element('polygon', points=' '.join(f'{a:g},{b:g}' for a, b in vertices), fill=color)


def text(label, x, y, size=17, color=INK, bold=False, anchor='start', max_width=None):
    font_file = 'arialbd.ttf' if bold else 'arial.ttf'
    font = ImageFont.truetype('C:/Windows/Fonts/' + font_file, round(size * 4))
    width = font.getlength(label) / 4
    if max_width is not None and width > max_width:
        raise ValueError(f'Text exceeds its allocation by {width - max_width:.1f} units: {label}')
    left = x - (width / 2 if anchor == 'middle' else width if anchor == 'end' else 0)
    if left < 0 or left + width > W:
        raise ValueError('Text is outside the artboard: ' + label)
    node = element('text', x=x, y=y, fill=color, font_family='Arial', font_size=size,
                   font_weight='700' if bold else '400', text_anchor=anchor)
    node.text = label
    TEXT_RECORDS.append(dict(text=label, x=x, y=y, size_pt=size * SCALE_PT,
                             approximate_width_pt=width * SCALE_PT))


def heading(letter, label, y):
    with group('panel_' + letter + '_heading'):
        text(letter, 20, y, 22, bold=True)
        text(label, 49, y, 20, bold=True, max_width=920)


def document(x, y, fill, stroke, width=23, height=29):
    fold = 6
    element('path', d=f'M{x},{y} H{x+width-fold} L{x+width},{y+fold} V{y+height} H{x} Z',
            fill=fill, stroke=stroke, stroke_width=1.2, stroke_linejoin='round')
    path([(x + width - fold, y), (x + width - fold, y + fold), (x + width, y + fold)], stroke, 1)
    path([(x + 5, y + 13), (x + width - 5, y + 13)], stroke, 1)
    path([(x + 5, y + 19), (x + width - 7, y + 19)], stroke, 1)


def lock(x, y, color, scale=1):
    element('path', d=f'M{x+3*scale},{y+8*scale} V{y+5*scale} '
            f'C{x+3*scale},{y-1*scale} {x+13*scale},{y-1*scale} {x+13*scale},{y+5*scale} V{y+8*scale}',
            fill='none', stroke=color, stroke_width=1.6)
    rect(x, y + 7 * scale, 16 * scale, 13 * scale, '#FFFFFF', color, 2 * scale, 1.4)
    circle(x + 8 * scale, y + 13 * scale, 1.4 * scale, color)
    path([(x + 8 * scale, y + 13 * scale), (x + 8 * scale, y + 17 * scale)], color, 1.1)


def network(x, y, width, height, color, large=False):
    columns = [[(x, y + height * k) for k in [0.18, 0.5, 0.82]],
               [(x + width * .5, y + height * k) for k in [.1, .5, .9]],
               [(x + width, y + height * k) for k in [.3, .7]]]
    for previous, following in zip(columns, columns[1:]):
        for start in previous:
            for end in following:
                path([start, end], '#B8CBD9', .85 if not large else 1.05)
    radius = 3.5 if large else 2.7
    for index, column in enumerate(columns):
        for point in column:
            circle(*point, radius, '#FFFFFF' if index == 0 else color, color, 1.05)


def token(label, x, y, width=65, height=27, size=18):
    fill, stroke, color = PALETTE[label]
    rect(x - width / 2, y - height / 2, width, height, fill, stroke, 5, 1.1)
    text(label, x, y + size * .34, size, color, True, 'middle', width - 6)


def panel_a():
    heading('a', 'Matched representation comparison', 27)
    with group('numerical_history'):
        text('Numerical history', 24, 58, 17.2, bold=True)
        path([(28, 108), (28, 74)], '#D1DBE4', 1)
        path([(28, 108), (154, 108)], '#D1DBE4', 1)
        points = [(29, 99), (43, 89), (65, 96), (75, 80), (112, 86), (132, 76), (153, 81)]
        path(points, BLUE, 2)
        for point in points:
            circle(*point, 2.7, '#FFFFFF', BLUE, 1.3)
    with group('numerical_bypass'):
        path([(163, 82), (720, 82), (720, 94)], BLUE, 1.8, arrow=True)
        text('Numerical inputs retained', 457, 70, 15.5, MUTED, anchor='middle')
    with group('available_text_history'):
        text('Available text', 24, 143, 17.2, bold=True)
        path([(28, 206), (161, 206)], LINE, 1.2, arrow=True)
        for x, payload in [(31, 'u'), (78, 'v'), (130, 'w')]:
            fill, stroke, _ = PALETTE[payload]
            document(x, 163, fill, stroke)
            path([(x + 11, 195), (x + 11, 210)], stroke, 1.2)
        text('Availability a ≤ q', 24, 227, 15.5, MUTED)
    with group('frozen_representation_configuration'):
        rect(203, 109, 218, 103, '#F3F7FB', '#A4BDD3', 9, 1.3)
        lock(216, 120, BLUE, .75)
        text('Frozen encoder', 323, 133, 18.4, BLUE, True, 'middle', 166)
        for index, (name, x) in enumerate([('GPT-2', 243), ('MiniLM', 311), ('FinBERT', 379)]):
            for row in range(3):
                for col in range(5):
                    values = ['#C4D8EA', '#92B3CF', '#E0EAF3', '#7199BA']
                    fill = values[(row + col + index) % len(values)]
                    rect(x - 23 + col * 9.3, 146 + row * 6.2, 7.2, 4.4, fill, radius=1)
            text(name, x, 179, 15.5, INK, True, 'middle', 70)
        for x in [277, 345]:
            path([(x, 145), (x, 184)], '#C9D8E5', .8)
        text('Pooling + truncation', 312, 204, 15.5, MUTED, anchor='middle', max_width=205)
        text('One encoder per configuration', 312, 232, 15.5, MUTED, anchor='middle', max_width=225)
        path([(164, 179), (195, 179)], BLUE, 1.7, arrow=True)
    with group('content_conditions'):
        rect(454, 111, 146, 102, '#FFFFFF', '#C5D1DB', 9, 1.25)
        for y, title, payloads in [(137, 'Aligned', ['u', 'v', 'w']),
                                   (169, 'Reassigned', ['w', 'u', 'v']),
                                   (200, 'Zero', ['0', '0', '0'])]:
            for index, payload in enumerate(payloads):
                circle(468 + index * 6, y - 5, 2, PALETTE[payload][1])
            text(title, 493, y, 16.4, max_width=101)
        path([(428, 179), (446, 179)], BLUE, 1.7, arrow=True)
    with group('shared_fusion_family'):
        rect(640, 94, 160, 104, '#F2F7F7', '#A0BCBE', 9, 1.3)
        text('Fusion model', 720, 118, 18.4, TEAL, True, 'middle')
        network(674, 132, 91, 39, TEAL, large=True)
        text('Trainable projection', 720, 190, 15.5, TEAL, anchor='middle', max_width=148)
        text('Selective · RecAvg · Attention', 720, 220, 15.5, MUTED, anchor='middle', max_width=229)
        path([(608, 179), (632, 179)], BLUE, 1.7, arrow=True)
    with group('paired_forecasts'):
        text('Paired forecasts', 911, 102, 17, bold=True, anchor='middle', max_width=148)
        rect(848, 115, 126, 82, '#FFFFFF', '#C5D1DB', 8, 1.2)
        path([(861, 180), (861, 127)], '#D1DBE4', 1)
        path([(861, 180), (963, 180)], '#D1DBE4', 1)
        path([(915, 125), (915, 180)], '#A8BAC9', 1, '3 3')
        path([(864, 165), (875, 147), (886, 155), (899, 139), (914, 146)], BLUE, 1.8)
        path([(915, 146), (931, 140), (947, 145), (962, 135)], ORANGE, 2)
        circle(962, 135, 2.4, '#FFFFFF', ORANGE, 1.3)
        text('q', 913, 193, 15.5, MUTED, anchor='middle')
        text('q+h', 955, 193, 15.5, MUTED, anchor='middle')
        path([(808, 154), (840, 154)], TEAL, 1.7, arrow=True)
        text('Same targets', 911, 220, 15.5, MUTED, anchor='middle', max_width=150)
    with group('matched_conditions_note'):
        text('Same fusion family, budget, paired seeds and forecast units',
             500, 254, 15.5, MUTED, anchor='middle', max_width=944)


def panel_b():
    heading('b', 'Content interventions on retained event slots', 291)
    rows = [(335, 'Aligned', ['u', 'v', 'w']),
            (376, 'Split shuffle', ['r', 'u', 's']),
            (417, 'Time-only', ['0', '0', '0']),
            (458, 'Within-history', ['v', 'w', 'u'])]
    with group('recipient_history_slots'):
        rect(291, 315, 596, 163, '#F7F9FB', radius=9)
        text('Same recipient event slots', 590, 313, 15.5, MUTED, anchor='middle')
        for x in [370, 548, 800]:
            path([(x, 321), (x, 475)], '#C8D5E0', 1, '2 4')
        for y, _, _ in rows:
            path([(308, y), (884, y)], LINE, 1.25)
        path([(903, 315), (903, 478)], INK, 1.8)
        text('Cutoff q', 943, 342, 17, bold=True, anchor='middle', max_width=95)
    for index, (y, label, payloads) in enumerate(rows):
        with group('intervention_' + str(index)):
            text(label, 25, y + 6, 18, bold=True, max_width=238)
            for x, payload in zip([370, 548, 800], payloads):
                token(payload, x, y)
    with group('donor_scope_note'):
        text('u, v, w are recipient-history vectors. r, s may come from other windows in the same entity/split.',
             25, 510, 15.5, MUTED, max_width=950)


def panel_c():
    heading('c', 'Model operation and known-signal calibration', 546)
    with group('condition_retrained_models'):
        rect(23, 562, 454, 145, '#F2F6FB', '#ABC2D6', 10, 1.3)
        text('Condition-retrained', 43, 589, 20, BLUE, True, max_width=410)
        for index, (label, color, fill) in enumerate([('A', BLUE, '#DDEAF5'),
                                                    ('S', ORANGE, '#F5E4D5'),
                                                    ('T', '#6F8190', '#E7EDF2')]):
            x = 48 + index * 140
            rect(x, 605, 29, 29, fill, color, 5, 1)
            text(label, x + 14.5, 625, 18, color, True, 'middle')
            path([(x + 36, 620), (x + 54, 620)], color, 1.4, arrow=True)
            network(x + 67, 607, 38, 26, color)
            text('θ' + label, x + 86, 654, 17.6, color, True, 'middle')
        text('Train and select each condition separately', 43, 678, 15.5, MUTED, max_width=412)
        text('SAG = 100(S − A)/S  ·  CBTG = 100(T − A)/T', 43, 698, 15.5, BLUE, max_width=414)
    with group('fixed_aligned_checkpoint'):
        rect(523, 562, 454, 145, '#F0F7F5', '#A6C1BA', 10, 1.3)
        text('Fixed aligned checkpoint', 543, 589, 20, TEAL, True, max_width=414)
        for y, label, payloads in [(618, 'Aligned', ['u', 'v', 'w']),
                                   (650, 'Historical', ['v', 'w', 'u'])]:
            text(label, 543, y + 5, 15.5, MUTED, max_width=75)
            for x, payload in zip([632, 655, 678], payloads):
                token(payload, x, y, width=18, height=22, size=15.5)
            path([(693, y), (712, y), (729, 632)], TEAL, 1.3)
        rect(733, 605, 71, 55, '#FFFFFF', '#98B7B3', 7, 1.15)
        network(747, 615, 42, 26, TEAL)
        lock(795, 599, TEAL, .73)
        text('θA', 768, 654, 16, TEAL, True, 'middle')
        path([(812, 632), (840, 632)], TEAL, 1.5, arrow=True)
        rect(846, 616, 105, 33, '#FFFFFF', '#A6C1BA', 6, 1.1)
        text('A vs H', 899, 638, 18, TEAL, True, 'middle')
        text('100(H − A)/H  ·  achieved content changes', 543, 698, 15.5, TEAL, max_width=414)
    with group('known_signal_generator'):
        rect(23, 735, 954, 51, '#F7F9FC', '#C6D2DF', 9, 1.1)
        circle(39, 752, 7, '#FFFFFF', BLUE, 1.3)
        path([(39, 748), (39, 752), (43, 754)], BLUE, 1.25)
        text('Known-signal calibration', 55, 757, 17.5, BLUE, True, max_width=280)
        text('Empirical clocks + MiniLM sentences', 43, 779, 15.5, MUTED, max_width=293)
        text('y = b(X) + λzc + ε', 371, 758, 19, INK, max_width=267)
        text('λ ∈ {0, 0.05, 0.1, 0.25, 1}', 371, 779, 15.5, MUTED, max_width=267)
        text('Both model operations', 688, 757, 17.5, BLUE, True, max_width=270)
        text('Capability × detection', 688, 779, 15.5, MUTED, max_width=270)
        path([(250, 733), (250, 710)], BLUE, 1.5, arrow=True)
        path([(750, 733), (750, 710)], TEAL, 1.5, arrow=True)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    with group('white_artboard'):
        rect(0, 0, W, H, '#FFFFFF')
    panel_a()
    panel_b()
    panel_c()
    output = HERE / 'fig1_mechanism.svg'
    ET.ElementTree(ROOT).write(output, encoding='utf-8', xml_declaration=True)
    report = dict(title='Representation-specific semantic audit', width_mm=WIDTH_MM,
                  height_mm=WIDTH_MM * H / W, minimum_font_pt=min(row['size_pt'] for row in TEXT_RECORDS),
                  text_blocks=len(TEXT_RECORDS), text_records=TEXT_RECORDS,
                  source_files={'manuscript\\sections\\method.tex': '5eb9bb0f52a0a470075d4df629586999dcaf316a7ce397283dddffafcdf12da7', 'manuscript\\sections\\design.tex': '7ab51b0f9c2bafa8fca9fc3d7fff33a8716dde666ea0ebc017e4ec6651276877'},
                  scientific_mapping={
                      'a': 'Matched natural-data representation configurations with retained numerical input and paired forecast units.',
                      'b': 'Entity/split reassignment, occurrence-preserving zeros and history-restricted reassignment use unchanged recipient slots.',
                      'c': 'Separate model selection and fixed aligned weights are evaluated on natural tasks and MiniLM known-signal worlds.',
                      'donor_law': 'Natural-data S uses split shuffle. Generated-world retraining uses historical rotation.',
                      'symbolic_inputs': 'u, v, w belong to a recipient history. r, s are illustrative split-level donors.',
                      'generator_baseline': 'b(X) = 0.8 v_last + 0.2 mean(v)',
                  })
    (HERE / 'design_manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    caption = (
        'Representation-specific semantic audit. (a) Frozen encoders define paired configurations. '
        'Pooling and truncation precede content controls, and the input projection is learned with '
        'the forecaster. (b) Recipient event slots are retained under split-level shuffle, zero '
        'content and within-history reassignment. Vectors u, v and w belong to the recipient history, '
        'while r and s may come from other windows in the same entity and split. The historical '
        'example illustrates a rotation. Historical controls comprise block rotation and nearest- '
        'vector swapping. (c) A/S/T models are trained and selected separately. Historical probes '
        'evaluate the selected aligned model at fixed weights, yielding losses A and H. Empirical- '
        'clock MiniLM worlds calibrate both operations against known signal strengths. Natural-data S '
        'uses split shuffle, and generated-world retraining uses within-history rotation. Validation '
        'capability and held-out detection characterize the calibrated response.\n'
    )
    (HERE / 'caption.txt').write_text(caption, encoding='utf-8')
    print(json.dumps(dict(svg=str(output), text_blocks=len(TEXT_RECORDS), minimum_font_pt=report['minimum_font_pt'])))


if __name__ == '__main__':
    main()
