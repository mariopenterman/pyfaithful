#include "pyc_code.h"
#include "pyc_module.h"
#include "data.h"

/* == Marshal structure for Code object ==
                1.0     1.3     1.5     2.1     2.3     3.0     3.8     3.11
argcount                short   short   short   long    long    long    long
posonlyargc                                                     long    long
kwonlyargc                                              long    long    long
nlocals                 short   short   short   long    long    long
stacksize                       short   short   long    long    long    long
flags                   short   short   short   long    long    long    long
code            Obj     Obj     Obj     Obj     Obj     Obj     Obj     Obj
consts          Obj     Obj     Obj     Obj     Obj     Obj     Obj     Obj
names           Obj     Obj     Obj     Obj     Obj     Obj     Obj     Obj
varnames                Obj     Obj     Obj     Obj     Obj     Obj
freevars                                Obj     Obj     Obj     Obj
cellvars                                Obj     Obj     Obj     Obj
locals+names                                                            Obj
locals+kinds                                                            Obj
filename        Obj     Obj     Obj     Obj     Obj     Obj     Obj     Obj
name            Obj     Obj     Obj     Obj     Obj     Obj     Obj     Obj
qualname                                                                Obj
firstline                       short   short   long    long    long    long
lntable                         Obj     Obj     Obj     Obj     Obj     Obj
exceptiontable                                                          Obj
*/

void PycCode::load(PycData* stream, PycModule* mod)
{
    if (mod->verCompare(1, 3) >= 0 && mod->verCompare(2, 3) < 0)
        m_argCount = stream->get16();
    else if (mod->verCompare(2, 3) >= 0)
        m_argCount = stream->get32();

    if (mod->verCompare(3, 8) >= 0)
        m_posOnlyArgCount = stream->get32();
    else
        m_posOnlyArgCount = 0;

    if (mod->majorVer() >= 3)
        m_kwOnlyArgCount = stream->get32();
    else
        m_kwOnlyArgCount = 0;

    if (mod->verCompare(1, 3) >= 0 && mod->verCompare(2, 3) < 0)
        m_numLocals = stream->get16();
    else if (mod->verCompare(2, 3) >= 0 && mod->verCompare(3, 11) < 0)
        m_numLocals = stream->get32();
    else
        m_numLocals = 0;

    if (mod->verCompare(1, 5) >= 0 && mod->verCompare(2, 3) < 0)
        m_stackSize = stream->get16();
    else if (mod->verCompare(2, 3) >= 0)
        m_stackSize = stream->get32();
    else
        m_stackSize = 0;

    if (mod->verCompare(1, 3) >= 0 && mod->verCompare(2, 3) < 0)
        m_flags = stream->get16();
    else if (mod->verCompare(2, 3) >= 0)
        m_flags = stream->get32();
    else
        m_flags = 0;

    if (mod->verCompare(3, 8) < 0) {
        // Remap flags to new values introduced in 3.8
        if (m_flags & 0xF0000000)
            throw std::runtime_error("Cannot remap unexpected flags");
        m_flags = (m_flags & 0xFFFF) | ((m_flags & 0xFFF0000) << 4);
    }

    m_code = LoadObject(stream, mod).cast<PycString>();
    m_consts = LoadObject(stream, mod).cast<PycSequence>();
    m_names = LoadObject(stream, mod).cast<PycSequence>();

    if (mod->verCompare(1, 3) >= 0)
        m_localNames = LoadObject(stream, mod).cast<PycSequence>();
    else
        m_localNames = new PycTuple;

    if (mod->verCompare(3, 11) >= 0)
        m_localKinds = LoadObject(stream, mod).cast<PycString>();
    else
        m_localKinds = new PycString;

    if (mod->verCompare(2, 1) >= 0 && mod->verCompare(3, 11) < 0)
        m_freeVars = LoadObject(stream, mod).cast<PycSequence>();
    else
        m_freeVars = new PycTuple;

    if (mod->verCompare(2, 1) >= 0 && mod->verCompare(3, 11) < 0)
        m_cellVars = LoadObject(stream, mod).cast<PycSequence>();
    else
        m_cellVars = new PycTuple;

    m_fileName = LoadObject(stream, mod).cast<PycString>();
    m_name = LoadObject(stream, mod).cast<PycString>();

    if (mod->verCompare(3, 11) >= 0)
        m_qualName = LoadObject(stream, mod).cast<PycString>();
    else
        m_qualName = new PycString;

    if (mod->verCompare(1, 5) >= 0 && mod->verCompare(2, 3) < 0)
        m_firstLine = stream->get16();
    else if (mod->verCompare(2, 3) >= 0)
        m_firstLine = stream->get32();

    if (mod->verCompare(1, 5) >= 0)
        m_lnTable = LoadObject(stream, mod).cast<PycString>();
    else
        m_lnTable = new PycString;

    if (mod->verCompare(3, 11) >= 0)
        m_exceptTable = LoadObject(stream, mod).cast<PycString>();
    else
        m_exceptTable = new PycString;
}

PycRef<PycString> PycCode::getCellVar(PycModule* mod, int idx) const
{
    if (mod->verCompare(3, 11) >= 0)
        return getLocal(idx);

    return (idx >= m_cellVars->size())
        ? m_freeVars->get(idx - m_cellVars->size()).cast<PycString>()
        : m_cellVars->get(idx).cast<PycString>();
}

int _parse_varint(PycBuffer& data, int& pos) {
    int b = data.getByte();
    pos += 1;

    int val = b & 0x3F;
    while (b & 0x40) {
        val <<= 6;

        b = data.getByte();
        pos += 1;

        val |= (b & 0x3F);
    }
    return val;
}

std::vector<PycExceptionTableEntry> PycCode::exceptionTableEntries() const
{
    PycBuffer data(m_exceptTable->value(), m_exceptTable->length());

    std::vector<PycExceptionTableEntry> entries;

    int pos = 0;
    while (!data.atEof()) {

        int start = _parse_varint(data, pos) * 2;
        int length = _parse_varint(data, pos) * 2;
        int end = start + length;
        
        int target = _parse_varint(data, pos) * 2;
        int dl = _parse_varint(data, pos);

        int depth = dl >> 1;
        bool lasti = bool(dl & 1);
        
        entries.push_back(PycExceptionTableEntry(start, end, target, depth, lasti));
    }

    return entries;
}

static unsigned int _parse_locvarint(PycBuffer& data)
{
    int read = data.getByte();
    unsigned int val = read & 0x3f;
    int shift = 0;
    while (read & 0x40) {
        read = data.getByte();
        shift += 6;
        val |= (unsigned int)(read & 0x3f) << shift;
    }
    return val;
}

static int _parse_locsvarint(PycBuffer& data)
{
    unsigned int uval = _parse_locvarint(data);
    return (uval & 1) ? -(int)(uval >> 1) : (int)(uval >> 1);
}

void PycCode::buildLineCache() const
{
    m_lineCacheBuilt = true;
    if (m_lnTable == nullptr || m_lnTable->length() == 0)
        return;
    PycBuffer data(m_lnTable->value(), m_lnTable->length());
    int line = m_firstLine;
    int cu = 0;
    while (!data.atEof()) {
        int first = data.getByte();
        if (!(first & 0x80))
            break;   // malformed: stop, matching the original early return
        int code = (first >> 3) & 0x0f;
        int length = (first & 0x07) + 1;
        int spanLine;
        int spanEndLine = -1;
        int scol = -1, ecol = -1;   // source columns (PEP 657 location table)
        switch (code) {
        case 15:
            spanLine = -1;
            break;
        case 14: {
            line += _parse_locsvarint(data);
            spanLine = line;
            spanEndLine = line + (int)_parse_locvarint(data);   // end-line delta
            int c1 = _parse_locvarint(data);        // start col + 1 (0 => unknown)
            int c2 = _parse_locvarint(data);        // end col + 1
            scol = c1 - 1;
            ecol = c2 - 1;
            break;
        }
        case 13:
            line += _parse_locsvarint(data);
            spanLine = line;
            break;
        case 12: case 11: case 10:
            line += code - 10;
            spanLine = line;
            scol = data.getByte();                  // start col
            ecol = data.getByte();                  // end col
            break;
        default: {
            // short form (code 0-9): one byte carries the low column bits and
            // the (end-start) column delta; the code is the column >> 3 group.
            int b = data.getByte();
            spanLine = line;
            scol = code * 8 + ((b >> 4) & 7);
            ecol = scol + (b & 0x0f);
            break;
        }
        }
        if (spanEndLine < 0)
            spanEndLine = spanLine;   // only long-form entries span lines
        m_lineCache.push_back({cu * 2, (cu + length) * 2, spanLine, spanEndLine, scol, ecol});
        cu += length;
    }
}

int PycCode::lineForOffset(int off) const
{
    if (!m_lineCacheBuilt)
        buildLineCache();
    // spans are non-overlapping and offset-ordered; binary search for the span
    // containing off (returns its line, which may be -1); -1 if in no span.
    int lo = 0, hi = (int)m_lineCache.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const LineSpan& s = m_lineCache[mid];
        if (off < s.start)
            hi = mid - 1;
        else if (off >= s.end)
            lo = mid + 1;
        else
            return s.line;
    }
    return -1;
}

int PycCode::colForOffset(int off) const
{
    if (!m_lineCacheBuilt)
        buildLineCache();
    int lo = 0, hi = (int)m_lineCache.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const LineSpan& s = m_lineCache[mid];
        if (off < s.start)
            hi = mid - 1;
        else if (off >= s.end)
            lo = mid + 1;
        else
            return s.scol;
    }
    return -1;
}

int PycCode::elineForOffset(int off) const
{
    if (!m_lineCacheBuilt)
        buildLineCache();
    int lo = 0, hi = (int)m_lineCache.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const LineSpan& s = m_lineCache[mid];
        if (off < s.start)
            hi = mid - 1;
        else if (off >= s.end)
            lo = mid + 1;
        else
            return s.eline;
    }
    return -1;
}

int PycCode::ecolForOffset(int off) const
{
    if (!m_lineCacheBuilt)
        buildLineCache();
    int lo = 0, hi = (int)m_lineCache.size() - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const LineSpan& s = m_lineCache[mid];
        if (off < s.start)
            hi = mid - 1;
        else if (off >= s.end)
            lo = mid + 1;
        else
            return s.ecol;
    }
    return -1;
}
